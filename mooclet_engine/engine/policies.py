import string

from numpy.random import choice, beta
from django.urls import reverse
from django.apps import apps
# from django.contrib.contenttypes.models import ContentType
from django.db.models import Avg, Sum
import json
from collections import Counter
from .utils.utils import sample_no_replacement
from django.db.models.query_utils import Q
import datetime
import numpy as np
from scipy.stats import invgamma
from django.forms.models import model_to_dict
import re
# arguments to policies:

# variables: list of variable objects, can be used to retrieve related data
# context: dict passed from view, contains current user, course, quiz, question context


def uniform_random(variables,context):
	return choice(context['mooclet'].version_set.all())

def uniform_random_time(variables,context):
	version = choice(context['mooclet'].version_set.all())
	version_dict = model_to_dict(version)
	if version_dict['text'] != '':
		dtnow = datetime.date.today()
		dtiso = dtnow.isoformat()
		version_dict['text'] = dtiso + ' ' + version_dict['text']
	return version_dict

def choose_policy_group(variables, context):
	Variable = apps.get_model('engine', 'Variable')
	Value = apps.get_model('engine', 'Value')
	Version = apps.get_model('engine', 'Version')
	Policy = apps.get_model('engine', 'Policy')

	var_name = str(context["mooclet"])+"_choose_policy_group"
	grp_var, created = Variable.objects.get_or_create(name=var_name)

	if "learner" not in context:
		return {"error": "please provide a learner ID"}

	if Value.objects.filter(variable=grp_var, learner=context["learner"]).exists():
		print("found prior policy")
		user_grp = Value.objects.filter(variable=grp_var, learner=context["learner"]).first()
		chosen_policy = user_grp.text
		print("prior policy: " + chosen_policy)
	else:
		print("selecting policy")
		policy_parameters = context["policy_parameters"].parameters
		policy_options = policy_parameters["policy_options"]
		print("options:")
		print(policy_options)
		policies = []
		weights = []
		for k, v in policy_options.items():
			policies.append(k)
			weights.append(v)
		chosen_policy = choice(policies, p=weights)
		print("chosen policy: " + chosen_policy)
		Value.objects.create(variable=grp_var, learner=context["learner"], text=chosen_policy)

	user_policy = Policy.objects.get(name=chosen_policy)
	version = user_policy.run_policy({"mooclet":context["mooclet"], "learner":context["learner"], "used_choose_group": True})

	if 'allowed_versions' in context:
		count = 1
		allowed_versions = context.get('allowed_versions', list(context['mooclet'].version_set.all()))
		if type(version) != dict:
			version_id = version.id
		else:
			version_id = version["id"]
  
		while (not allowed_versions.filter(pk=version_id).exists()) and count < context.get("maximum_allowed", 20):
			count += 1
			if Value.objects.filter(variable=grp_var, learner=context["learner"]).exists():
				print("found prior policy")
				user_grp = Value.objects.filter(variable=grp_var, learner=context["learner"]).first()
				chosen_policy = user_grp.text
				print("prior policy: " + chosen_policy)
			else:
				print("selecting policy")
				policy_parameters = context["policy_parameters"].parameters
				policy_options = policy_parameters["policy_options"]
				print("options:")
				print(policy_options)
				policies = []
				weights = []
				for k, v in policy_options.items():
					policies.append(k)
					weights.append(v)
				chosen_policy = choice(policies, p=weights)
				print("chosen policy: " + chosen_policy)
				Value.objects.create(variable=grp_var, learner=context["learner"], text=chosen_policy)

			user_policy = Policy.objects.get(name=chosen_policy)
			version = user_policy.run_policy({"mooclet":context["mooclet"], "learner":context["learner"], "used_choose_group": True})
			if type(version) != dict:
				version_id = version.id
			else:
				version_id = version["id"]

		if not allowed_versions.filter(pk=version_id).exists() and count == context.get("maximum_allowed", 20):
			print("RANDOM CHOOSE")
			version = choice(context['mooclet'].version_set.all())

	if type(version) != dict:
		version_dict = model_to_dict(version)
	else:
		version_dict = version
	version_dict["policy"] = chosen_policy
	version_dict["policy_id"] = user_policy.id
	return version_dict

def choose_mooclet_group(variables, context):
	Variable = apps.get_model('engine', 'Variable')
	Value = apps.get_model('engine', 'Value')
	Mooclet = apps.get_model('engine', 'Mooclet')

	var_name = str(context["mooclet"])+"_choose_mooclet_group"
	grp_var, created = Variable.objects.get_or_create(name=var_name)

	if "learner" not in context:
		return {"error": "please provide a learner ID"}
	else:
		if Value.objects.filter(variable=grp_var, learner=context["learner"]).exists():
			print("found prior mooclet")
			user_grp = Value.objects.filter(variable=grp_var, learner=context["learner"]).first()
			user_group = user_grp.text
			print("prior mooclet: " + user_group)
			user_mooclet = Mooclet.objects.get(name=user_group)
			version = user_mooclet.run(context={"learner":context["learner"]})
			if type(version) != dict:
				version_dict = model_to_dict(version)
			else:
				version_dict = version
			return version_dict
		else:
			print("selecting mooclet")
			policy_parameters = context["policy_parameters"].parameters
			mooclet_options = policy_parameters["mooclet_options"]
			print("options:")
			print(mooclet_options)
			mooclets = []
			weights = []
			for k, v in mooclet_options.items():
				mooclets.append(k)
				weights.append(v)
			chosen_mooclet = choice(mooclets, p=weights)
			print("chosen mooclet: " + chosen_mooclet)
			Value.objects.create(variable=grp_var, learner=context["learner"], text=chosen_mooclet)
			usr_mooclet = Mooclet.objects.get(name=chosen_mooclet)
			version = usr_mooclet.run(context={"learner":context["learner"]})
			if type(version) != dict:
				version_dict = model_to_dict(version)
			else:
				version_dict = version
			return version_dict

def choose_user_level_group(variables, context):
    Variable = apps.get_model('engine', 'Variable')
    Value = apps.get_model('engine', 'Value')
    Version = apps.get_model('engine', 'Version')
    Policy = apps.get_model('engine', 'Policy')
    PolicyParams = apps.get_model('engine', 'PolicyParameters')
    Mooclet = apps.get_model('engine', 'Mooclet')
    
    var_name = str(context["mooclet"])+"_choose_user_level_group"
    grp_var, created = Variable.objects.get_or_create(name=var_name)
    
    if "learner" not in context:
        return {"error": "please provide a learner ID"}
    
    mooclet_values = Value.objects.filter(learner=context["learner"]).filter(~Q(variable=grp_var))
    mooclet_arms = Version.objects.filter(mooclet=context['mooclet'])
    policy_parameters = context["policy_parameters"].parameters         
    policies = list(policy_parameters["policies"])
    reward_name = policy_parameters["reward"]
    context_names = list(policy_parameters["contexts"])

    if Value.objects.filter(variable=grp_var, learner=context["learner"]).exists():
        print("found prior mooclet")
        user_grp = Value.objects.filter(variable=grp_var, learner=context["learner"]).first()
        user_group = user_grp.text
        print("prior mooclet: " + user_group)
        user_mooclet = Mooclet.objects.get(name=user_group)
    else:
        print("initialize new mooclet")

        # copy user values to new mooclet
        new_mooclet_name = context["mooclet"].name + "_" + context["learner"].name
        user_mooclet = Mooclet.objects.create(name=new_mooclet_name)
        print("new mooclet: {}".format(user_mooclet.name))

        # create new arms
        # Note: assume sub-mooclets share the same arm name as the current mooclet
        new_user_arms = mooclet_arms.all()
        for arm in new_user_arms:
            arm.pk = None
            arm.mooclet = user_mooclet
            arm.save()
        print("new user arms: {}".format([arm for arm in new_user_arms]))

        # create new policies
        for index, policy_param in enumerate(policies):
            policy = Policy.objects.get(name=policy_param.get('name'))
            parameters = policy_param.get('parameters')
            
            if policy.name in ['ts_configurable', 'ts_configurable_with_r']:
                posteriors_data = parameters.get('current_posteriors', None)
                if posteriors_data:
                    arm_keys = list(posteriors_data.keys())
                    for arm_key in arm_keys:
                        arm_value = posteriors_data[arm_key]
                        mooclet_arm = Version.objects.get(pk=int(arm_key))
                        if mooclet_arm:
                            new_user_arm = Version.objects.get(name=mooclet_arm.name)
                            posteriors_data[new_user_arm] = arm_value
                        del posteriors_data[arm_key]

            PolicyParams.objects.create(
                mooclet=user_mooclet,
                policy=policy, 
                parameters=policy_param.get('parameters'),
                latest_update=datetime.datetime.now()
            )
            
            print("policy created: {}".format(policy.name))
            
            # link the first policy to new mooclet
            if index == 0:
                user_mooclet.policy = policy
                user_mooclet.save()
        
        # Create history record to link current mooclet and sub-mooclet
        Value.objects.create(variable=grp_var, learner=context["learner"], text=new_mooclet_name)
    
    # update mooclet id to all current values
    mooclet_values.update(mooclet=user_mooclet)
    print("policy value updated: {}".format(mooclet_values))
    
    # Note: assume sub-mooclets share the same arm name as the current mooclet
    for arm in mooclet_arms:
        version_values = mooclet_values.filter(variable__name='version', version=arm)
        reward_values = mooclet_values.filter(variable__name=reward_name, version=arm)
        user_arm = Version.objects.get(mooclet=user_mooclet, name=arm.name)

        # update arm id to all version values
        version_values.update(version=user_arm)

        # update arm id to all reward values
        reward_values.update(version=user_arm)
        
        print("arm updated: {}".format(arm.pk))
    
    version = user_mooclet.run(context={"learner":context["learner"]})
    if type(version) != dict:
        version_dict = model_to_dict(version)
    else:
        version_dict = version
    return version_dict

def weighted_random(variables,context):
    all_versions = context['mooclet'].version_set.all()
    policy_parameters = context["policy_parameters"].parameters
    prob_dist = policy_parameters["probability_distribution"]
    weights = [prob_dist[version.name] if version.name in prob_dist else 0.0 for version in all_versions]
    return choice(all_versions, p=weights)


def weighted_random_time(variables, context):
	print("started_wrt")
	Value = apps.get_model('engine', 'Value')
	Weight = variables.get(name='version_weight')
	weight_data = Value.objects.filter(variable=Weight, version__in=context['mooclet'].version_set.all())
	versions = [weight.version for weight in weight_data]
	weights = [weight.value for weight in weight_data]
	version = choice(versions, p=weights)
	version_dict = model_to_dict(version)
	if version_dict['text'] != '':
		dtnow = datetime.date.today()
		dtiso = dtnow.isoformat()
		version_dict['text'] = dtiso + ' ' + version_dict['text']
	return version_dict

def thompson_sampling_placeholder(variables,context):
	return choice(context['mooclet'].version_set.all())

def thompson_sampling(variables,context):
	versions = context['mooclet'].version_set.all()
	#import models individually to avoid circular dependency
	Variable = apps.get_model('engine', 'Variable')
	Value = apps.get_model('engine', 'Value')
	Version = apps.get_model('engine', 'Version')
	# version_content_type = ContentType.objects.get_for_model(Version)
	#priors we set by hand - will use instructor rating and confidence in future
	# TODO : all explanations are having the same prior.

	# context is the following json :
	#   {
	#   'policy_parameters':
	#       {
	#       'outcome_variable_name':<name of the outcome variable',
	#       'max_rating': <maximum value of the outcome variable>,
	#       'prior':
	#           {'success':<prior success value>},
	#           {'failure':<prior failure value>},
	#       }
	#   }
	policy_parameters = context["policy_parameters"].parameters

	prior_success = policy_parameters['prior']['success']

	prior_failure = policy_parameters['prior']['failure']
	outcome_variable_name = policy_parameters['outcome_variable_name']
	#max value of version rating, from qualtrics
	max_rating = policy_parameters['max_rating']

	version_to_show = None
	max_beta = 0

	for version in versions:
		if "used_choose_group" in context and context["used_choose_group"] == True:
			student_ratings = Variable.objects.get(name=outcome_variable_name).get_data(context={'version': version, 'mooclet': context['mooclet'], 'policy': 'thompson_sampling'})
		else:
			student_ratings = Variable.objects.get(name=outcome_variable_name).get_data(context={'version': version, 'mooclet': context['mooclet']})
		if student_ratings:
			student_ratings = student_ratings.all()
			# student_ratings is a pandas.core.series.Series variable
			rating_count = student_ratings.count()
			rating_average = student_ratings.aggregate(Avg('value'))
			rating_average = rating_average['value__avg']
			if rating_average is None:
				rating_average = 0

		else:
			rating_average = 0
			rating_count = 0
		#get instructor conf and use for priors later
		#add priors to db
		# prior_success_db, created = Variable.objects.get_or_create(name='thompson_prior_success')
		# prior_success_db_value = Value.objects.filter(variable=prior_success_db, version=version).last()
		# if prior_success_db_value:
		# 	#there is already a value, so update it
		# 	prior_success_db_value.value = prior_success
		# 	prior_success_db_value.save()
		# else:
		# 	#no db value
		# 	prior_success_db_value = Value.objects.create(variable=prior_success_db, version=version, value=prior_success)

		# prior_failure_db, created = Variable.objects.get_or_create(name='thompson_prior_failure')
		# prior_failure_db_value = Value.objects.filter(variable=prior_failure_db, version=version).last()
		# if prior_failure_db_value:
		# 	#there is already a value, so update it
		# 	prior_failure_db_value.value = prior_failure
		# 	prior_failure_db_value.save()
		# else:
		# 	#no db value
		# 	prior_failure_db_value = Value.objects.create(variable=prior_failure_db, version=version, value=prior_failure)


		#TODO - log to db later?
		successes = (rating_average * rating_count) + prior_success
		failures = (max_rating * rating_count) - (rating_average * rating_count) + prior_failure
		print("successes: " + str(successes))
		print("failures: " + str(failures))
		version_beta = beta(successes, failures)

		if version_beta > max_beta:
			max_beta = version_beta
			version_to_show = version

	return version_to_show


def sample_without_replacement(variables, context):
	mooclet = context['mooclet']
	policy_parameters = context['policy_parameters']
	# print "parameters:"
	# print policy_parameters
	conditions = None
	#print "starting"
	previous_versions = None

	Variable = apps.get_model('engine', 'Variable')
	Value = apps.get_model('engine', 'Value')
	Version = apps.get_model('engine', 'Version')

	if policy_parameters:
		# print "Has policy parameters"
		policy_parameters = policy_parameters.parameters

		if policy_parameters["type"] == "per-user" and context["learner"]:
			# print "Per user and Has learner"
			previous_versions = Version.objects.filter(value__variable__name="version", value__learner=context["learner"], mooclet=mooclet).all()
			# previous_versions = Value.objects.filter(learner=context['learner'], mooclet=mooclet,
			# 					variable__name="version").values_list("version", flat=True)

		if 'variables' in policy_parameters and previous_versions:
			# print "previous versions " + str(len(previous_versions))
			variables = policy_parameters['variables']
			values = Value.objects.filter(version__in=previous_versions, variable__name__in=variables.keys()).select_related('variable','version').all()
			value_list = []
			for version in previous_versions:
				# print list(values.filter(version=version).all().values())
				value_list = value_list + list(values.filter(version=version).all().values("text","variable__name"))
			# print value_list
			conditions = {}
			for variable in variables.keys():
				#var_values = value_list.filter(variable__name=variable).values_list("text", flat=True)
				var_values = list(filter(lambda x: x["variable__name"] == variable, value_list))
				var_values = list(map(lambda x: x["text"], var_values))

				conditions[variable] = sample_no_replacement(variables[variable], var_values)


		elif 'variables' in policy_parameters:
			# print "variables but no user or prior context"
			#user hasn't seen versions previously
			variables = policy_parameters['variables']
			conditions = {}
			for variable in variables.items():
				conditions[variable[0]] = choice(variable[1])


		# elif policy_parameters["type"] == "per-user" and context["learner"] and 'variables' not in policy_parameters:
		# 	if previous_versions:
		# 		version = sample_no_replacement(mooclet.version_set.all(), previous_versions)
		# 	else:
		# 		version = choice(context['mooclet'].version_set.all())

		if conditions:
			# print "conditions"
			# print conditions
			all_versions = mooclet.version_set.all()
			correct_versions = all_versions

			for condition in conditions:
				correct_versions = correct_versions.all().filter(Q(value__variable__name=condition, value__text=conditions[condition]))

			# print("All versions len:"+str(len(correct_versions.all())))
			version = correct_versions.first()



		else:
			#no version features, do random w/o replace within the versions
			# print "nothing"
			all_versions = mooclet.version_set.all()#values_list("version", flat=True)
			if not previous_versions:
				print ("no prev vers")
				previous_versions = Version.objects.filter(value__variable__name="version", mooclet=mooclet).all()
			if previous_versions:
				version = sample_no_replacement(all_versions, previous_versions)
			else:
				version = choice(all_versions)


	else:
		#no version features, do random w/o replace within the versions
		# print "nothing"
		all_versions = mooclet.version_set.all()#values_list("version", flat=True)
		if not previous_versions:
			previous_versions = Version.objects.filter(value__variable__name="version", mooclet=mooclet).all()#Value.objects.filter(mooclet=mooclet, variable__name="version").values_list("version", flat=True)
		if previous_versions:
			version = sample_no_replacement(all_versions, previous_versions)
		else:
			version = choice(all_versions)

	return version




def sample_without_replacement2(variables, context):
	mooclet = context['mooclet']
	policy_parameters = context['policy_parameters']
	print ("parameters:")
	print (policy_parameters)
	conditions = None
	print ("starting")
	previous_versions = None
	version = None

	Variable = apps.get_model('engine', 'Variable')
	Value = apps.get_model('engine', 'Value')
	Version = apps.get_model('engine', 'Version')

	if policy_parameters:
		print ("Has policy parameters")
		policy_parameters = policy_parameters.parameters

		if policy_parameters["type"] == "per-user" and context["learner"]:
			print ("Per user and Has learner")
			#ALL versions
			previous_versions = Version.objects.filter(value__variable__name="version", mooclet=mooclet).all()

			previous_versions_user = previous_versions.filter(value__learner=context["learner"])

			if bool(previous_versions_user):
				print ("has_previous_versions")
				#if previous versions, return a new set of factors
				all_versions = mooclet.version_set.all()
				factor_names = policy_parameters["variables"].keys()
				previous_version_factors = Value.objects.filter(variable__name__in=factor_names, version__in=previous_versions_user)

				new_factors = all_versions.exclude(value__variable__name__in=factor_names, value__text__in=previous_version_factors.values_list('text',flat=True)).all()
				version = choice(new_factors)
				#pass

			elif bool(previous_versions):
				min_seen_version = None
				min_seen_count = 0
				#no previous versions for the given user
				for version in previous_versions:
					count = Value.objects.filter(variable__name="version", version=version).count()
					if min_seen_version is None:
						min_seen_version = version
						min_seen_count = count
					elif count < min_seen_count:
						min_seen_version = version
						min_seen_count = count
				version = min_seen_version

			else:
				all_versions = mooclet.version_set.all()
				version = choice(all_versions)






	return version

def if_then_rules(variables, context):
	Variable = apps.get_model('engine', 'Variable')
	Value = apps.get_model('engine', 'Value')
	Version = apps.get_model('engine', 'Version')

	policy_parameters = context['policy_parameters']
	parameters = policy_parameters.parameters

	for case in parameters:
		print(case)
		if case != "else":
			logical_statement = parameters[case]['logical_statement']
			print(f"{'-' * 12} IF THEN RULES {'-' * 12}")
			print("logical statement: ")
			print(logical_statement)
			#flag = False
			# chunk = ''
			# replace_variables = []
			# for i in logical_statement.split():
			# 	if flag:
			# 		if "}" in i:
			# 			chunk = chunk + i
			# 			replace_variables.append(chunk)
			# 			flag = False
			# 		else:
			# 			chunk = chunk + i
			# 	if "{" in i and "}" in i:
			# 		replace_variables.append(i)
			# 	elif "{" in i and "}" not in i:
			# 		flag = True
			# 		chunk = chunk + i

			replace_variables = re.findall(r"{.*?}", logical_statement)
			print("replace variables: ")
			print(replace_variables)
			logical_statement_clean = logical_statement
			var_dict = {}
			print("learner name: ")
			print(context['learner'].name)
			for variable in replace_variables:
				#format {var_name|mooclet=mooclet|version=version}
				query_args = {}
				variable_name = variable.strip('{}')
				items = variable_name.split('|')
				variable_name = items[0]
				query_args['variable__name'] = variable_name
				for item in items[1:]:
					item = item.split('=')
					if "__name" not in item[0]:
						item[1] = int(item[1])
					query_args[item[0]] = item[1]
				query_args['learner'] = context['learner']

				logical_statement_clean = logical_statement_clean.replace(variable, "{{{}}}".format(variable_name))
				print("variable name: ")
				print(variable_name)
				# if variable == 'version|mooclet':
				# 	pass

				#var_db = Variable.objects.get(name=variable)
				#should we always get the first? In practice this means most recently added
				val = Value.objects.filter(**query_args).first()
				if val:
					var_dict[variable_name] = val.value
				else:
					var_dict[variable_name] = None
			print("var dict:")
			print(var_dict)
			print("logical statement clean:")
			print(logical_statement_clean)
			logical_converted = logical_statement_clean.format(**var_dict)
			print("logical converted: ")
			print(logical_converted)
			truth = None
			try:
				truth = eval(logical_converted)
			except:
				pass #what should we do in this case (evaluation of statement fails)?
			if truth:
				print("case: ")
				print(case)
				weights = parameters[case]['probability_distribution']
				version_name = choice(list(weights.keys()), p=list(weights.values()))
				version = Version.objects.get(name=version_name, mooclet=context['mooclet'])
				return version
			else:
				pass
	if 'else' in parameters:
		weights = parameters['else']
		version_name = choice(list(weights.keys()), p=list(weights.values()))
		version = Version.objects.get(name=version_name)
		return version

	# Randomize between arms if no condition is reached.
	version_to_show = choice(context['mooclet'].version_set.all())
	if_then_rules_ur, created = Variable.objects.get_or_create(name="IF_THEN_RULES_UR")
	Value.objects.create(
		variable=if_then_rules_ur, 
		value=1.0, 
		text="IF_THEN_RULES_UR", 
		learner=context["learner"], 
		mooclet=context["mooclet"], 
		version=version_to_show
	)
	return version_to_show

# Runs exact same policy as above, but makes soure the output is a valid date

def if_then_rules_time(variables, context):
	version = if_then_rules(variables, context)
	version_dict = model_to_dict(version)
	if version_dict['text'] != '':
		dtnow = datetime.date.today()
		dtiso = dtnow.isoformat()
		version_dict['text'] = dtiso + ' ' + version_dict['text']
	return version_dict



# Draw thompson sample of (reg. coeff., variance) and also select the optimal action
def thompson_sampling_contextual(variables, context):
	'''
	thompson sampling policy with contextual information.
	Outcome is estimated using bayesian linear regression implemented by NIG conjugate priors.
	map dict to version
	get the current user's context as a dict
	'''
	Variable = apps.get_model('engine', 'Variable')
	Value = apps.get_model('engine', 'Value')
	Version = apps.get_model('engine', 'Version')
	# Store normal-inverse-gamma parameters
	policy_parameters = context['policy_parameters']
	parameters = policy_parameters.parameters
	print(str(parameters))
	# Store regression equation string
	regression_formula = parameters['regression_formula']
	# Action space, assumed to be a json
	action_space = parameters['action_space']

	# Include intercept can be true or false
	include_intercept = parameters['include_intercept']

	# Store contextual variables
	contextual_vars = parameters['contextual_variables']
	print("contextual_vars_original: " + str(contextual_vars))
	if 'learner' not in context:
		pass
	print('learner' + str(context['learner']))
	contextual_vars = Value.objects.filter(variable__name__in=contextual_vars, learner=context['learner'])
	contextual_vars_dict = {}
	for val in contextual_vars:
		contextual_vars_dict[val.variable.name] = val.value
		contextual_vars = contextual_vars_dict
	print('contextual vars: ' + str(contextual_vars))
	current_enrolled = Value.objects.filter(variable__name="version", mooclet=context["mooclet"],
											policy__name="thompson_sampling_contextual").count()
	if "uniform_threshold" in parameters:
		uniform_threshold = parameters["uniform_threshold"]
	# number of current participants within uniform random threshold, random sample
	if "uniform_threshold" in parameters and current_enrolled <= parameters["uniform_threshold"]:
		ur_or_ts, created = Variable.objects.get_or_create(name="UR_or_TSCONTEXTUAL")
		version_to_show = choice(context['mooclet'].version_set.all())
		Value.objects.create(variable=ur_or_ts, value=0.0,
							 text="UR_COLDSTART", learner=context["learner"], mooclet=context["mooclet"],
							 version=version_to_show)
		version_dict = model_to_dict(version_to_show)
		version_dict["selection_method"] = "uniform_random_coldstart"
		return version_dict
	# Get current priors parameters (normal-inverse-gamma)
	mean = parameters['coef_mean']
	cov = parameters['coef_cov']
	variance_a = parameters['variance_a']
	variance_b = parameters['variance_b']
	print('prior mean: ' + str(mean))
	print('prior cov: ' + str(cov))
	# Draw variance of errors
	precesion_draw = invgamma.rvs(variance_a, 0, variance_b, size=1)
	# Draw regression coefficients according to priors
	coef_draw = np.random.multivariate_normal(mean, precesion_draw * cov)
	print('sampled coeffs: ' + str(coef_draw))
	## Generate all possible action combinations
	# Initialize action set
	all_possible_actions = [{}]

	# Itterate over actions label names
	for cur in action_space:
		# Store set values corresponding to action labels
		cur_options = action_space[cur]

	# Initialize list of feasible actions
		new_possible = []
	# Itterate over action set
		for a in all_possible_actions:
		# Itterate over value sets correspdong to action labels
			for cur_a in cur_options:
				new_a = a.copy()
				new_a[cur] = cur_a

		# Check if action assignment is feasible
				if is_valid_action(new_a):
			# Append feasible action to list
					new_possible.append(new_a)
					all_possible_actions = new_possible

	filtered = filter(lambda x: sum(list(x.values())) <= 1, all_possible_actions)
	all_possible_actions = list(filtered)
	# Print entire action set
	print('all possible actions: ' + str(all_possible_actions))

	## Calculate outcome for each action and find the best action
	best_outcome = -np.inf
	best_action = None

	print('regression formula: ' + regression_formula)
	# Itterate of all feasible actions
	for action in all_possible_actions:
		independent_vars = action.copy()
		independent_vars.update(contextual_vars)
		print('independent vars: ' + str(independent_vars))
		# Compute expected reward given action
		outcome = calculate_outcome(independent_vars,coef_draw, include_intercept, regression_formula)
		print("curr_outcome" + str(outcome))
		print('outcome: ' + str(best_outcome))
		# Keep track of optimal (action, outcome)
		if best_action is None or outcome > best_outcome:
			best_outcome = outcome
			best_action = action
	# Print optimal action
	print('best action: ' + str(best_action))

	version_to_show = Version.objects.filter(mooclet=context['mooclet'])

	version_to_show = version_to_show.get(version_json__contains=best_action)

	if "precesion_draw" in parameters and parameters["precesion_draw"] == 1:
		precesion, created = Variable.objects.get_or_create(name="precesion_draw")
		Value.objects.create(variable=precesion, value=precesion_draw,
							 text=str(precesion_draw), learner=context["learner"], mooclet=context["mooclet"],
							 version=version_to_show)

	if "coef_draw" in parameters and parameters["coef_draw"] == 1:
		coef_sample, created = Variable.objects.get_or_create(name="coef_draw")
		Value.objects.create(variable=coef_sample, value=0.0,
							 text=str(coef_draw), learner=context["learner"], mooclet=context["mooclet"],
							 version=version_to_show)

	#TODO: convert best action into version
	#version_to_show = {}
	return version_to_show

# Draw thompson sample of (reg. coeff., variance) and also select the optimal action
def thompson_sampling_contextual_group(variables, context):
	'''
	thompson sampling policy with contextual information.
	Outcome is estimated using bayesian linear regression implemented by NIG conjugate priors.
	map dict to version
	get the current user's context as a dict
	'''

	Variable = apps.get_model('engine', 'Variable')
	Value = apps.get_model('engine', 'Value')
	Version = apps.get_model('engine', 'Version')
	# Store normal-inverse-gamma parameters
	policy_parameters = context['policy_parameters']
	parameters = policy_parameters.parameters

	# Store regression equation string
	regression_formula = parameters['regression_formula']

	# Action space, assumed to be a json
	action_space = parameters['action_space']

	# Include intercept can be true or false
	include_intercept = parameters['include_intercept']

	# Store contextual variables
	contextual_vars = parameters['contextual_variables']
	if 'learner' not in context:
		pass
	contextual_vars = Value.objects.filter(variable__name__in=contextual_vars, learner=context['learner'])
	contextual_vars_dict = {}
	for val in contextual_vars:
		contextual_vars_dict[val.variable.name] = val.value
	contextual_vars = contextual_vars_dict
	print('contextual vars: ' + str(contextual_vars))

	# Get current priors parameters (normal-inverse-gamma)
	mean = parameters['coef_mean']
	cov = parameters['coef_cov']
	variance_a = parameters['variance_a']
	variance_b = parameters['variance_b']
	print('prior mean: ' + str(mean))
	print('prior cov: ' + str(cov))
	# Draw variance of errors
	# TODO set to 1 for normal distribution
	precesion_draw = invgamma.rvs(variance_a, 0, variance_b, size=1)

	# Draw regression coefficients according to priors
	coef_draw = np.random.multivariate_normal(mean, precesion_draw * cov)
	print('sampled coeffs: ' + str(coef_draw))

	## Generate all possible action combinations
	# Initialize action set
	all_possible_actions = [{}]

	# Itterate over actions label names
	for cur in action_space:

			# Store set values corresponding to action labels
		cur_options = action_space[cur]

			# Initialize list of feasible actions
		new_possible = []

			# Itterate over action set
		for a in all_possible_actions:

				# Itterate over value sets correspdong to action labels
			for cur_a in cur_options:
				new_a = a.copy()
				new_a[cur] = cur_a

					# Check if action assignment is feasible
				if is_valid_action(new_a):

						# Append feasible action to list
					new_possible.append(new_a)
					all_possible_actions = new_possible

	# Print entire action set
	print('all possible actions: ' + str(all_possible_actions))

	## Calculate outcome for each action and find the best action
	best_outcome = -np.inf
	best_action = None

	print('regression formula: ' + regression_formula)
	# Itterate of all feasible actions
	for action in all_possible_actions:
		independent_vars = action.copy()
		independent_vars.update(contextual_vars)
		print('independent vars: ' + str(independent_vars))
		# Compute expected reward given action
		outcome = calculate_outcome(independent_vars,coef_draw, include_intercept, regression_formula)
		print("curr_outcome" + str(outcome))
		print('outcome: ' + str(best_outcome))
		# Keep track of optimal (action, outcome)
		if best_action is None or outcome > best_outcome:
			best_outcome = outcome
			best_action = action

	# Print optimal action
	print('best action: ' + str(best_action))
	version_to_show = Version.objects.filter(mooclet=context['mooclet'])

	version_to_show = version_to_show.filter(version_json__contains=best_action)

	version_to_show = choice(version_to_show)

	#TODO: convert best action into version
	#version_to_show = {}
	return version_to_show

# Compute expected reward given context and action of user
# Inputs: (design matrix row as dict, coeff. vector, intercept, reg. eqn.)
def calculate_outcome(var_dict, coef_list, include_intercept, formula):
	'''
	:param var_dict: dict of all vars (actions + contextual) to their values
	:param coef_list: coefficients for each term in regression
	:param include_intercept: whether intercept is included
	:param formula: regression formula
	:return: outcome given formula, coefficients and variables values
	'''
	# Strip blank beginning and end space from equation
	formula = formula.strip()

	# Split RHS of equation into variable list (context, action, interactions)
	vars_list = list(map(str.strip, formula.split('~')[1].strip().split('+')))


	# Add 1 for intercept in variable list if specified
	if include_intercept:
		vars_list.insert(0,1.)

	# Raise assertion error if variable list different length then coeff list
	print(vars_list)
	print(coef_list)
	assert(len(vars_list) == len(coef_list))

	# Initialize outcome
	outcome = 0.

	dummy_loops = 0
	for k in range(20):
		dummy_loops += 1
	print(dummy_loops)

	print(str(type(coef_list)))
	print(np.shape(coef_list))
	coef_list = coef_list.tolist()
	print("coef list length: " + str(len(coef_list)))
	print("vars list length: " + str(len(vars_list)))
	print("vars_list " + str(vars_list))
	print("curr_coefs " + str(coef_list))

	## Use variables and coeff list to compute expected reward
	# Itterate over all (var, coeff) pairs from regresion model
	num_loops = 0
	for j in range(len(coef_list)): #var, coef in zip(vars_list,coef_list):
		var = vars_list[j]
		coef = coef_list[j]
		## Determine value in variable list
		# Initialize value (can change in loop)
		value = 1.
		# Intercept has value 1
		if type(var) == float:
			value = 1.

		# Interaction term value
		elif '*' in var:
			interacting_vars = var.split('*')

			interacting_vars = list(map(str.strip,interacting_vars))
			# Product of variable values in interaction term
			for i in range(0, len(interacting_vars)):
				value *= var_dict[interacting_vars[i]]
		# Action or context value
		else:
			value = var_dict[var]

		# Compute expected reward (hypothesized regression model)
		print("value " + str(value) )
		print("coefficient " + str(coef))
		outcome += coef * value
		num_loops += 1
		print("loop number: " + str(num_loops))

	print("Number of loops: " + str(num_loops))
	return outcome

# Check whether action is feasible (only one level of the action variables can be realized)
def is_valid_action(action):
	'''
	checks whether an action is valid, meaning, no more than one vars under same category are assigned 1
	'''

	# Obtain labels for each action
	keys = action.keys()

	# Itterate over each action label
	for cur_key in keys:

			# Find the action labels with multiple levels
		if '_' not in cur_key:
			continue
		value = 0
		prefix = cur_key.rsplit('_',1)[0] + '_'

			# Compute sum of action variable with multiple levels
		for key in keys:
			if key.startswith(prefix):
				value += action[key]
			# Action not feasible if sum of indicators is more than 1
		if value > 1:
			return False

	# Return true if action is valid
	return True


# Posteriors for beta and variance
def posteriors(y, X, m_pre, V_pre, a1_pre, a2_pre):
  #y = list of uotcomes
  #X = design matrix
  #priors input by users, but if no input then default
  #m_pre vector 0 v_pre is an identity matrix - np.identity(size of params) a1 & a2 both 2. save the updates
  #get the reward as a spearate vector. figure ut batch size issues (time based)

  # Data size

  datasize = len(y)

  # X transpose
  Xtranspose = np.matrix.transpose(X)

  # Residuals
  # (y - Xb) and (y - Xb)'
  resid = np.subtract(y, np.dot(X,m_pre))
  resid_trans = np.matrix.transpose(resid)

  # N x N middle term for gamma update
  # (I + XVX')^{-1}
  mid_term = np.linalg.inv(np.add(np.identity(datasize), np.dot(np.dot(X, V_pre),Xtranspose)))

  ## Update coeffecients priors

  # Update mean vector
  # [(V^{-1} + X'X)^{-1}][V^{-1}mu + X'y]
  m_post = np.dot(np.linalg.inv(np.add(np.linalg.inv(V_pre), np.dot(Xtranspose,X))), np.add(np.dot(np.linalg.inv(V_pre), m_pre), np.dot(Xtranspose,y)))

  # Update covariance matrix
  # (V^{-1} + X'X)^{-1}
  V_post = np.linalg.inv(np.add(np.linalg.inv(V_pre), np.dot(Xtranspose,X)))

  ## Update precesion prior

  # Update gamma parameters
  # a + n/2 (shape parameter)
  a1_post = a1_pre + datasize/2

  # b + (1/2)(y - Xmu)'(I + XVX')^{-1}(y - Xmu) (scale parameter)
  a2_post = a2_pre + (np.dot(np.dot(resid_trans, mid_term), resid))/2

  ## Posterior draws

  # Precesions from inverse gamma (shape, loc, scale, draws)
  precesion_draw = invgamma.rvs(a1_post, 0, a2_post, size = 1)

  # Coeffecients from multivariate normal
  beta_draw = np.random.multivariate_normal(m_post, precesion_draw*V_post)

  # List with beta and s^2
  #beta_s2 = np.append(beta_draw, precesion_draw)

  # Return posterior drawn parameters
  # output: [(betas, s^2, a1, a2), V]
  return{"coef_mean": m_post,
		"coef_cov": V_post,
		"variance_a": a1_post,
		"variance_b": a2_post}

  #return [np.append(np.append(beta_s2, a1_post), a2_post), V_post]



TSPOSTDIFF_THRESH = 0.15 # Threshold for TS PostDiff

def thompson_sampling_postdiff(variables,context):
	"""
	Assumes only 2 versions
	"""
	versions = context['mooclet'].version_set.all()
	print("versions")
	print(versions)
	#import models individually to avoid circular dependency
	Variable = apps.get_model('engine', 'Variable')
	Value = apps.get_model('engine', 'Value')
	Version = apps.get_model('engine', 'Version')
	# version_content_type = ContentType.objects.get_for_model(Version)
	#priors we set by hand - will use instructor rating and confidence in future
	# TODO : all explanations are having the same prior.

	# context is the following json :
	#   {
	#   'policy_parameters':
	#       {
	#       'outcome_variable_name':<name of the outcome variable',
	#       'max_rating': <maximum value of the outcome variable>,
	#       'prior':
	#           {'success':<prior success value>},
	#           {'failure':<prior failure value>},
	#       }
	#   }
	policy_parameters = context["policy_parameters"].parameters

	prior_success = policy_parameters['prior']['success']

	tspostdiff_thresh = policy_parameters["tspostdiff_thresh"]
	prior_failure = policy_parameters['prior']['failure']
	outcome_variable_name = policy_parameters['outcome_variable_name']
	#max value of version rating, from qualtrics
	max_rating = policy_parameters['max_rating']

	version_to_show = None
	max_beta = 0
	version_to_draw_dict = {} # {"version1": 0.5, "version2": 0.4}

	for version in versions:
		print(version.name)
		if "used_choose_group" in context and context["used_choose_group"] == True:
			student_ratings = Variable.objects.get(name=outcome_variable_name).get_data(context={'version': version, 'mooclet': context['mooclet'], 'policy': 'thompson_sampling_postdiff'})
		else:
			student_ratings = Variable.objects.get(name=outcome_variable_name).get_data(context={'version': version, 'mooclet': context['mooclet']})
		if student_ratings:
			student_ratings = student_ratings.all()
			# student_ratings is a pandas.core.series.Series variable
			rating_count = student_ratings.count()
			rating_average = student_ratings.aggregate(Avg('value'))
			rating_average = rating_average['value__avg']
			if rating_average is None:
				rating_average = 0

		else:
			rating_average = 0
			rating_count = 0

		successes = (rating_average * rating_count) + prior_success
		failures = (max_rating * rating_count) - (rating_average * rating_count) + prior_failure
		print("successes: " + str(successes))
		print("failures: " + str(failures))
		#version_beta = beta(successes, failures) #Not used
		version_to_draw_dict[version] = (successes, failures)
		print(version_to_draw_dict)


	#	if version_beta > max_beta:
	#		max_beta = version_beta
	#		version_to_show = version
	version_beta_1 = beta(version_to_draw_dict.values()[0][0], version_to_draw_dict.values()[0][1])
	version_beta_2 = beta(version_to_draw_dict.values()[1][0], version_to_draw_dict.values()[1][1])

	diff = abs(version_beta_1 - version_beta_2)

	#log whether was chosen by ts or ur
	ur_or_ts, created = Variable.objects.get_or_create(name="UR_or_TS")

	if diff < tspostdiff_thresh:# do UR
		print("choices to show")
		print(context['mooclet'].version_set.all())
		version_to_show = choice(context['mooclet'].version_set.all())
		Value.objects.create(variable=ur_or_ts, value=0.0,
							text="UR", learner=context["learner"], mooclet=context["mooclet"],
							version=version_to_show)
		version_dict = model_to_dict(version_to_show)
		version_dict["selection_method"] = "uniform_random"
		return version_dict

	else: #Do TS with resampling
		for version in version_to_draw_dict.keys():
			successes = version_to_draw_dict[version][0]
			failures = version_to_draw_dict[version][1]
			version_beta = beta(successes, failures)
			print("pre max beta: " +str(max_beta))
			print("version_beta: " + str(version_beta))
			if version_beta > max_beta:
				max_beta = version_beta
				version_to_show = version


		#log policy chosen
		Value.objects.create(variable=ur_or_ts, value=1.0,
							text="TS", learner=context["learner"], mooclet=context["mooclet"],
							version=version_to_show)

		version_dict = model_to_dict(version_to_show)
		version_dict["selection_method"] = "thompson_sampling"
		return version_dict


def thompson_sampling_uniform_start(variables,context):
	versions = context['mooclet'].version_set.all()
	#import models individually to avoid circular dependency
	Variable = apps.get_model('engine', 'Variable')
	Value = apps.get_model('engine', 'Value')
	Version = apps.get_model('engine', 'Version')
	# version_content_type = ContentType.objects.get_for_model(Version)
	#priors we set by hand - will use instructor rating and confidence in future
	# TODO : all explanations are having the same prior.

	# context is the following json :
	#   {
	#   'policy_parameters':
	#       {
	#       'outcome_variable_name':<name of the outcome variable',
	#       'max_rating': <maximum value of the outcome variable>,
	#       'prior':
	#           {'success':<prior success value>},
	#           {'failure':<prior failure value>},
	#       }
	#   }
	policy_parameters = context["policy_parameters"].parameters

	prior_success = policy_parameters['prior']['success']

	prior_failure = policy_parameters['prior']['failure']
	outcome_variable_name = policy_parameters['outcome_variable_name']
	#max value of version rating, from qualtrics
	max_rating = policy_parameters['max_rating']

	uniform_threshold = policy_parameters["uniform_threshold"]
	n_enrolled = Value.objects.filter(variable__name="version", mooclet=context["mooclet"], policy__name="thompson_sampling_uniform_start").count()

	ur_or_ts, created = Variable.objects.get_or_create(name="UR_or_TS")
	if n_enrolled <= uniform_threshold:
		version_to_show = choice(context['mooclet'].version_set.all())
		Value.objects.create(variable=ur_or_ts, value=0.0,
							text="UR", learner=context["learner"], mooclet=context["mooclet"],
							version=version_to_show)
		version_dict = model_to_dict(version_to_show)
		version_dict["selection_method"] = "uniform_random"
		return version_dict

	else:
		version_to_show = None
		max_beta = 0

		for version in versions:
			if "used_choose_group" in context and context["used_choose_group"] == True:
				student_ratings = Variable.objects.get(name=outcome_variable_name).get_data(context={'version': version, 'mooclet': context['mooclet'], 'policy': 'thompson_sampling_uniform_start'})
			else:
				student_ratings = Variable.objects.get(name=outcome_variable_name).get_data(context={'version': version, 'mooclet': context['mooclet']})
			if student_ratings:
				student_ratings = student_ratings.all()
				# student_ratings is a pandas.core.series.Series variable
				rating_count = student_ratings.count()
				rating_average = student_ratings.aggregate(Avg('value'))
				rating_average = rating_average['value__avg']
				if rating_average is None:
					rating_average = 0

			else:
				rating_average = 0
				rating_count = 0


			#TODO - log to db later?
			successes = (rating_average * rating_count) + prior_success
			failures = (max_rating * rating_count) - (rating_average * rating_count) + prior_failure
			print("successes: " + str(successes))
			print("failures: " + str(failures))
			version_beta = beta(successes, failures)

			if version_beta > max_beta:
				max_beta = version_beta
				version_to_show = version

		Value.objects.create(variable=ur_or_ts, value=1.0,
							text="TS", learner=context["learner"], mooclet=context["mooclet"],
							version=version_to_show)
		version_dict = model_to_dict(version_to_show)
		version_dict["selection_method"] = "thompson_sampling"
		return version_dict
		#return version_to_show


def thompson_sampling_batched(variables,context):
	versions = context['mooclet'].version_set.all()
	#import models individually to avoid circular dependency
	#no problem if missing data is random, but could
	#introduce bias if rewards are dont being sent is biased
	#suppose the reason they don't send reward is because they hate it and close survey
	#this skews for the subsequent batches. _could_ correct itself if we have the rewards come later
	#but this is also a general problem
	#note that using bernoulli for initial draws can be kind of problematic
	#because it's noisy and can result in imbalanced initial distributions
	#so we have more data about some arms than others
	#so maybe we could for batch 1 do evenly distributed as much as possible?
	Variable = apps.get_model('engine', 'Variable')
	Value = apps.get_model('engine', 'Value')
	Version = apps.get_model('engine', 'Version')
	PolicyParametersHistory = apps.get_model('engine', 'PolicyParametersHistory')
	# version_content_type = ContentType.objects.get_for_model(Version)
	#priors we set by hand - will use instructor rating and confidence in future
	# TODO : all explanations are having the same prior.

	# context is the following json :
	#   {
	#   'policy_parameters':
	#       {
	#       'outcome_variable_name':<name of the outcome variable',
	#       'max_rating': <maximum value of the outcome variable>,
	#       'prior':
	#           {'success':<prior success value>},
	#           {'failure':<prior failure value>},
	#       }
	#   }
	policy_parameters = context["policy_parameters"].parameters

	prior_success = policy_parameters['prior']['success']

	prior_failure = policy_parameters['prior']['failure']
	outcome_variable_name = policy_parameters['outcome_variable_name']
	#max value of version rating, from qualtrics
	max_rating = policy_parameters['max_rating']

	batch_size = policy_parameters["batch_size"]
	current_enrolled = Value.objects.filter(variable__name="version", mooclet=context["mooclet"], policy__name="thompson_sampling_batched").count()

	if "current_posteriors" not in policy_parameters or current_enrolled % batch_size == 0:
		#update policyparameters
		current_posteriors = {}
		for version in versions:
			if "used_choose_group" in context and context["used_choose_group"] == True:
				student_ratings = Variable.objects.get(name=outcome_variable_name).get_data(context={'version': version, 'mooclet': context['mooclet'], 'policy': 'thompson_sampling_batched'})
			else:
				student_ratings = Variable.objects.get(name=outcome_variable_name).get_data(context={'version': version, 'mooclet': context['mooclet']})
			if student_ratings:
				student_ratings = student_ratings.all()
				# student_ratings is a pandas.core.series.Series variable
				rating_count = student_ratings.count()
				rating_average = student_ratings.aggregate(Avg('value'))
				rating_average = rating_average['value__avg']
				if rating_average is None:
					rating_average = 0


			else:
				rating_average = 0
				rating_count = 0


			#TODO - log to db later?
			successes = (rating_average * rating_count)
			failures = (max_rating * rating_count) - (rating_average * rating_count)
			current_posteriors[version.id] = {"successes":successes, "failures": failures}


		new_history = PolicyParametersHistory.create_from_params(context["policy_parameters"])

		new_history.save()
		context["policy_parameters"].parameters["current_posteriors"] = current_posteriors
		new_update_time = datetime.datetime.now()
		context["policy_parameters"].latest_update = new_update_time
		context["policy_parameters"].save()
	else:
		current_posteriors = policy_parameters["current_posteriors"]



	version_to_show = None
	max_beta = 0

	for version in current_posteriors:
		#one issue we have when analyzing data curerntly is we need to know how the updates are done

		version_beta = beta(current_posteriors[version]["successes"] + prior_success, current_posteriors[version]["failures"] + prior_failure)

		if version_beta > max_beta:
			max_beta = version_beta
			version_to_show = Version.objects.get(id=version)

	# Value.objects.create(variable=ur_or_ts, value=1.0,
	# 					text="TS", learner=context["learner"], mooclet=context["mooclet"],
	# 					version=version_to_show)
	# version_dict = model_to_dict(version_to_show)
	# version_dict["selection_method"] = "thompson_sampling"
	return version_to_show
		#return version_to_show

def ts_configurable(variables, context):
	versions = context['mooclet'].version_set.all()
	#import models individually to avoid circular dependency
	#no problem if missing data is random, but could
	#introduce bias if rewards are dont being sent is biased
	#suppose the reason they don't send reward is because they hate it and close survey
	#this skews for the subsequent batches. _could_ correct itself if we have the rewards come later
	#but this is also a general problem
	#note that using bernoulli for initial draws can be kind of problematic
	#because it's noisy and can result in imbalanced initial distributions
	#so we have more data about some arms than others
	#so maybe we could for batch 1 do evenly distributed as much as possible?
	Variable = apps.get_model('engine', 'Variable')
	Value = apps.get_model('engine', 'Value')
	Version = apps.get_model('engine', 'Version')
	PolicyParametersHistory = apps.get_model('engine', 'PolicyParametersHistory')
	# version_content_type = ContentType.objects.get_for_model(Version)
	#priors we set by hand - will use instructor rating and confidence in future
	# TODO : all explanations are having the same prior.

	# context is the following json :
	#   {
	#   'policy_parameters':
	#       {
	#       'outcome_variable_name':<name of the outcome variable',
	#       'max_rating': <maximum value of the outcome variable>,
	#       'prior':
	#           {'success':<prior success value>},
	#           {'failure':<prior failure value>},
	#       }
	#   }
	policy_parameters = context["policy_parameters"].parameters

	prior_success = policy_parameters['prior']['success']

	prior_failure = policy_parameters['prior']['failure']
	outcome_variable_name = policy_parameters['outcome_variable_name']
	#max value of version rating, from qualtrics
	min_rating, max_rating = policy_parameters["min_rating"] if "min_rating" in policy_parameters else 0, policy_parameters['max_rating']

	epsilon = policy_parameters['epsilon'] if 'epsilon' in policy_parameters else 0

	if "batch_size" in policy_parameters:
		batch_size = policy_parameters["batch_size"]
	if "uniform_threshold" in policy_parameters:
		uniform_threshold = policy_parameters["uniform_threshold"]
	current_enrolled = Value.objects.filter(variable__name="version", mooclet=context["mooclet"], policy__name="ts_configurable").count()

	#number of current participants within uniform random threshold, random sample
	if "uniform_threshold" in policy_parameters and current_enrolled <= policy_parameters["uniform_threshold"]:
		ur_or_ts, created = Variable.objects.get_or_create(name="UR_or_TS")
		version_to_show = choice(context['mooclet'].version_set.all())
		Value.objects.create(variable=ur_or_ts, value=0.0,
							text="UR_COLDSTART", learner=context["learner"], mooclet=context["mooclet"],
							version=version_to_show)
		version_dict = model_to_dict(version_to_show)
		version_dict["selection_method"] = "uniform_random_coldstart"
		return version_dict

	#number of current participants within uniform random threshold, random sample
	if np.random.uniform() < epsilon:
		ur_or_ts, created = Variable.objects.get_or_create(name="UR_or_TS")
		version_to_show = choice(context['mooclet'].version_set.all())
		Value.objects.create(variable=ur_or_ts, value=0.0,
							text="EPSILON", learner=context["learner"], mooclet=context["mooclet"],
							version=version_to_show)
		version_dict = model_to_dict(version_to_show)
		version_dict["selection_method"] = "uniform_random_coldstart"
		return version_dict

	if "current_posteriors" not in policy_parameters or current_enrolled % batch_size == 0 :
		#update policyparameters
		current_posteriors = {}
		for version in versions:
			if "used_choose_group" in context and context["used_choose_group"] == True:
				student_ratings = Variable.objects.get(name=outcome_variable_name).get_data(context={'version': version, 'mooclet': context['mooclet'], 'policy': 'ts_configurable'})
			else:
				student_ratings = Variable.objects.get(name=outcome_variable_name).get_data(context={'version': version, 'mooclet': context['mooclet']})
			if student_ratings:
				student_ratings = student_ratings.all()
				# student_ratings is a pandas.core.series.Series variable
				rating_count = student_ratings.count()
				# rating_average = student_ratings.aggregate(Avg('value'))
				sum_rewards = student_ratings.aggregate(Sum('value'))
				sum_rewards = sum_rewards['value__sum']
				# rating_average = rating_average['value__avg']
				# if rating_average is None:
					# rating_average = 0
			else:
				# rating_average = 0
				rating_count = 0
				sum_rewards = 0


			success_update = (sum_rewards - rating_count * min_rating) / (max_rating - min_rating)
			successes = success_update
			failures = rating_count - success_update

			current_posteriors[version.id] = {"successes":successes, "failures": failures}


		new_history = PolicyParametersHistory.create_from_params(context["policy_parameters"])

		new_history.save()
		context["policy_parameters"].parameters["current_posteriors"] = current_posteriors
		new_update_time = datetime.datetime.now()
		context["policy_parameters"].latest_update = new_update_time
		context["policy_parameters"].save()
	else:
		current_posteriors = policy_parameters["current_posteriors"]

	version_dict = {}
	for version in current_posteriors:
		current_posteriors[version]["successes"]
		version_dict[Version.objects.get(id=version)] = {"successes":
														current_posteriors[version]["successes"] + prior_success,
														"failures":  current_posteriors[version]["failures"] + prior_failure}
	print(version_dict)
	if "tspostdiff_thresh" in policy_parameters:
		print("ts_postdiff")
		return ts_postdiff_sample(policy_parameters["tspostdiff_thresh"], version_dict, context)
	else:
		return ts_sample(version_dict, context)


def ts_postdiff_sample(tspostdiff_thresh, versions_dict, context):
	"""
	Inputs are a threshold and a dict of versions to successes and failures e.g.:
	{version1: {successess: 1, failures: 1}.
	version2: {successess: 1, failures: 1}, ...}
	"""
	Variable = apps.get_model('engine', 'Variable')
	Value = apps.get_model('engine', 'Value')
	Version = apps.get_model('engine', 'Version')


	version_values = list(versions_dict.values())
	version_beta_1 = beta(version_values[0]["successes"], version_values[0]["failures"])
	version_beta_2 = beta(version_values[1]["successes"], version_values[1]["failures"])

	diff = abs(version_beta_1 - version_beta_2)

	#log whether was chosen by ts or ur
	ur_or_ts, created = Variable.objects.get_or_create(name="UR_or_TS")

	if diff < tspostdiff_thresh:# do UR
		# #print("choices to show")
		#print(context['mooclet'].version_set.all())
		version_to_show = list(versions_dict.keys())
		version_to_show = choice(version_to_show)
		#version_to_show = Version.objects.get(id=version_to_show)
		Value.objects.create(variable=ur_or_ts, value=0.0,
							text="UR", learner=context["learner"], mooclet=context["mooclet"],
							version=version_to_show)
		version_dict = model_to_dict(version_to_show)
		version_dict["selection_method"] = "uniform_random"
		return version_dict

	else: #Do TS with resampling
		version_to_show = None
		max_beta = 0
		for version in versions_dict.keys():
			successes = versions_dict[version]["successes"]
			failures = versions_dict[version]["failures"]
			version_beta = beta(successes, failures)
			#print("pre max beta: " +str(max_beta))
			#print("version_beta: " + str(version_beta))
			if version_beta > max_beta:
				max_beta = version_beta
				version_to_show = version


		#log policy chosen
		Value.objects.create(variable=ur_or_ts, value=1.0,
							text="TS", learner=context["learner"], mooclet=context["mooclet"],
							version=version_to_show)

		version_dict = model_to_dict(version_to_show)
		version_dict["selection_method"] = "thompson_sampling_postdiff"
		return version_dict


def ts_sample(versions_dict, context):
	"""
	Input is a dict of versions to successes and failures e.g.:
	{version1: {successess: 1, failures: 1}.
	version2: {successess: 1, failures: 1}, ...}, and context
	"""
	Variable = apps.get_model('engine', 'Variable')
	Value = apps.get_model('engine', 'Value')
	Version = apps.get_model('engine', 'Version')
	version_to_show = None
	max_beta = 0
	for version in versions_dict.keys():
		successes = versions_dict[version]["successes"]
		failures = versions_dict[version]["failures"]
		version_beta = beta(successes, failures)
		#print("pre max beta: " +str(max_beta))
		#print("version_beta: " + str(version_beta))
		if version_beta > max_beta:
			max_beta = version_beta
			version_to_show = version

	ur_or_ts, created = Variable.objects.get_or_create(name="UR_or_TS")

	Value.objects.create(variable=ur_or_ts, value=1.0,
							text="TS_NONPOSTDIFF", learner=context["learner"], mooclet=context["mooclet"],
							version=version_to_show)
	version_dict = model_to_dict(version_to_show)
	version_dict["selection_method"] = "ts_nonpostdiff"
	#version_to_show = Version.objects.get(id=version_to_show)
	return version_to_show


def ts_configurable_with_r(variables, context):
	import rpy2.robjects as R

	versions = context['mooclet'].version_set.all()
	Variable = apps.get_model('engine', 'Variable')
	Value = apps.get_model('engine', 'Value')
	Version = apps.get_model('engine', 'Version')
	PolicyParametersHistory = apps.get_model('engine', 'PolicyParametersHistory')
	PolicyParameter = apps.get_model('engine', 'PolicyParameters')
	policy_parameters = context["policy_parameters"].parameters
	prior_success = policy_parameters['prior']['success']
	prior_failure = policy_parameters['prior']['failure']
	outcome_variable_name = policy_parameters['outcome_variable_name']
	min_rating, max_rating = policy_parameters["min_rating"] if "min_rating" in policy_parameters else 0, policy_parameters['max_rating']

	# Batch
	if "batch_size" in policy_parameters:
		batch_size = policy_parameters["batch_size"]
	else:
		batch_size = 1

	if "uniform_threshold" in policy_parameters:
		uniform_threshold = policy_parameters["uniform_threshold"]
	else:
		uniform_threshold = 0

	current_enrolled = Value.objects.filter(
		variable__name="version", 
		mooclet=context["mooclet"], 
		policy__name="ts_configurable_with_r"
	)
	
	if uniform_threshold > 0 and current_enrolled.count() <= policy_parameters["uniform_threshold"]:
		ur_or_ts, created = Variable.objects.get_or_create(name="UR_or_TS")
		version_to_show = choice(context['mooclet'].version_set.all())
		Value.objects.create(variable=ur_or_ts, value=0.0,
							text="UR_COLDSTART", learner=context["learner"], mooclet=context["mooclet"],
							version=version_to_show)
		version_dict = model_to_dict(version_to_show)
		version_dict["selection_method"] = "uniform_random_coldstart"
		return version_dict

	policy = PolicyParameter.objects.get(mooclet=context["mooclet"], policy__name='ts_configurable_with_r')
	if policy.latest_update:
		new_enrolled = current_enrolled.filter(timestamp__gte=policy.latest_update)
	else:
		new_enrolled = current_enrolled

	if "current_posteriors" not in policy_parameters or new_enrolled.count() >= batch_size:
		#update policyparameters
		current_posteriors = {}
		for version in versions:
			if "used_choose_group" in context and context["used_choose_group"] == True:
				student_ratings = Variable.objects.get(name=outcome_variable_name).get_data(context={'version': version, 'mooclet': context['mooclet'], 'policy': 'ts_configurable_with_r'})
			else:
				student_ratings = Variable.objects.get(name=outcome_variable_name).get_data(context={'version': version, 'mooclet': context['mooclet']})
			if student_ratings:
				student_ratings = student_ratings.all()
				rating_count = student_ratings.count()
				sum_rewards = student_ratings.aggregate(Sum('value'))
				sum_rewards = sum_rewards['value__sum']
			else:
				rating_count = 0
				sum_rewards = 0
    
			update_para = R.r(
				r'''
				function(para){
					success <- para[1]
					failure <- para[2]
					total_num_reward <- para[3]
					sum_reward <- para[4]

					success <- success + sum_reward
					failure <- failure + (total_num_reward - sum_reward)
					return(para=c(success, failure))
				}
				'''
			)
   
			as_numeric = R.r['as.numeric']
			para_vec = R.FloatVector((0, 0, rating_count, sum_rewards))
			para = update_para(para_vec)
			para = as_numeric(para)

			successes, failures = tuple(para)
			current_posteriors[version.id] = {"successes":successes, "failures": failures}

		new_history = PolicyParametersHistory.create_from_params(context["policy_parameters"])
		new_history.save()

		context["policy_parameters"].parameters["current_posteriors"] = current_posteriors
		new_update_time = datetime.datetime.now()
		context["policy_parameters"].latest_update = new_update_time
		context["policy_parameters"].save()
	else:
		current_posteriors = policy_parameters["current_posteriors"]

	version_dict = {}
	for version in current_posteriors:
		version_dict[Version.objects.get(id=version)] = {
      		"successes": current_posteriors[version]["successes"] + prior_success, 
        	"failures":  current_posteriors[version]["failures"] + prior_failure
        }

	print(version_dict)
	if "toptwots_thresh" in policy_parameters:
		print("toptwo_ts")
		return toptwo_ts_sample_with_r(policy_parameters["toptwots_thresh"], version_dict, context)
	elif "tsprobclip_thresh" in policy_parameters:
		print("ts_probclip")
		return ts_probclip_sample_with_r(policy_parameters["tsprobclip_thresh"], version_dict, context)
	elif "tspostdiff_thresh" in policy_parameters:
		print("ts_postdiff")
		return ts_postdiff_sample_with_r(policy_parameters["tspostdiff_thresh"], version_dict, context)
	else:
		return ts_sample_with_r(version_dict, context)


def toptwo_ts_sample_with_r(toptwots_thresh, versions_dict, context):
	"""
	Inputs are a threshold and a dict of versions to successes and failures e.g.:
	{version1: {successess: 1, failures: 1}.
	version2: {successess: 1, failures: 1}, ...}
	"""
	import rpy2.robjects as R

	Variable = apps.get_model('engine', 'Variable')
	Value = apps.get_model('engine', 'Value')

	version_success_failure = list(versions_dict.items())
	arm1_success_failure, arm2_success_failure = version_success_failure[0], version_success_failure[1]
	success_arm1, failure_arm1 = arm1_success_failure[1]["successes"], arm1_success_failure[1]["failures"]
	success_arm2, failure_arm2 = arm2_success_failure[1]["successes"], arm2_success_failure[1]["failures"]

	toptwo_ts = R.r(
		r'''
		function(para){
			sa <- para[1]
			fa <- para[2]
			sb <- para[3]
			fb <- para[4]
			algorithm_para <- para[5]
			draw_a <- rbeta(1, sa, fa)
			draw_b <- rbeta(1, sb, fb)
			
			is_ur <- 0
			if (runif(1) < algorithm_para){
				draw_a <- runif(1)
				draw_b <- runif(1)
				is_ur <- 1
			} 
			arm <- 1
			if(draw_a < draw_b){
				arm <- 2
			}
			return(c(arm, is_ur))
		}
		'''
    )

	as_numeric = R.r['as.numeric']
	para = R.FloatVector((success_arm1, failure_arm1, success_arm2, failure_arm2, toptwots_thresh))
	assignment = toptwo_ts(para)
	assignment = as_numeric(assignment)
	arm, is_ur = tuple(assignment)

	if arm == 1:
		# Arm 1 chosen
		version_to_show = arm1_success_failure[0]
	else:
		# Arm 2 chosen
		version_to_show = arm2_success_failure[0]

	#log whether was chosen by ts or ur
	ur_or_ts, created = Variable.objects.get_or_create(name="UR_or_TS")
	if is_ur == 1:
		Value.objects.create(
      		variable=ur_or_ts, value=0.0, text="UR", 
            learner=context["learner"], mooclet=context["mooclet"],
			version=version_to_show
   		)
		version_dict = model_to_dict(version_to_show)
		version_dict["selection_method"] = "uniform_random"
		return version_dict
	else:
		#log policy chosen
		Value.objects.create(
      		variable=ur_or_ts, value=1.0, text="TS", 
            learner=context["learner"], mooclet=context["mooclet"],
			version=version_to_show
		)
		version_dict = model_to_dict(version_to_show)
		version_dict["selection_method"] = "toptwo_thompson_sampling"
		return version_dict


def ts_probclip_sample_with_r(tsprobclip_thresh, versions_dict, context):
	"""
	Inputs are a threshold and a dict of versions to successes and failures e.g.:
	{version1: {successess: 1, failures: 1}.
	version2: {successess: 1, failures: 1}, ...}
	"""
	import rpy2.robjects as R

	version_success_failure = list(versions_dict.items())
	arm1_success_failure, arm2_success_failure = version_success_failure[0], version_success_failure[1]
	success_arm1, failure_arm1 = arm1_success_failure[1]["successes"], arm1_success_failure[1]["failures"]
	success_arm2, failure_arm2 = arm2_success_failure[1]["successes"], arm2_success_failure[1]["failures"]

	ts_probclip = R.r(
		r'''
		function(para){
			sa <- para[1]
			fa <- para[2]
			sb <- para[3]
			fb <- para[4]
			algorithm_para <- para[5]
			draw_a <- rbeta(1, sa, fa)
			draw_b <- rbeta(1, sb, fb)
		
			a <- rbeta(1000, sa, fa)
			b <- rbeta(1000, sb, fb)
			pab <- mean(a > b)
			
			draw_a <- median(c(algorithm_para, 1 - algorithm_para, pab))
			
			arm <- 1
			if(draw_a < draw_b) {
				arm <- 2
			}
			return(arm)
		}
		'''
    )

	as_numeric = R.r['as.numeric']
	para = R.FloatVector((success_arm1, failure_arm1, success_arm2, failure_arm2, tsprobclip_thresh))
	assignment = ts_probclip(para)
	assignment = as_numeric(assignment)
	arm = assignment[0]

	if arm == 1:
		# Arm 1 chosen
		version_to_show = arm1_success_failure[0]
	else:
		# Arm 2 chosen
		version_to_show = arm2_success_failure[0]

	version_dict = model_to_dict(version_to_show)
	version_dict["selection_method"] = "thompson_sampling_probclip"
	return version_dict


def ts_postdiff_sample_with_r(tspostdiff_thresh, versions_dict, context):
	"""
	Inputs are a threshold and a dict of versions to successes and failures e.g.:
	{version1: {successess: 1, failures: 1}.
	version2: {successess: 1, failures: 1}, ...}
	"""
	import rpy2.robjects as R

	Variable = apps.get_model('engine', 'Variable')
	Value = apps.get_model('engine', 'Value')

	version_success_failure = list(versions_dict.items())
	arm1_success_failure, arm2_success_failure = version_success_failure[0], version_success_failure[1]
	success_arm1, failure_arm1 = arm1_success_failure[1]["successes"], arm1_success_failure[1]["failures"]
	success_arm2, failure_arm2 = arm2_success_failure[1]["successes"], arm2_success_failure[1]["failures"]

	ts_postdiff = R.r(
		r'''
		function(para){
			sa <- para[1]
			fa <- para[2]
			sb <- para[3]
			fb <- para[4]
			c <- para[5]
			draw_a <- rbeta(1, sa, fa)
			draw_b <- rbeta(1, sb, fb)

			is_ur <- 0
			if (abs(draw_a - draw_b) < c){
				draw_a <- runif(1)
				draw_b <- runif(1)
				is_ur <- 1
			} else {
				draw_a <- rbeta(1, sa, fa)
				draw_b <- rbeta(1, sb, fb)
			}

			arm <- 1
			if(draw_a < draw_b){
				arm <- 2
			}
			return(c(arm, is_ur))
		}
		'''
    )

	as_numeric = R.r['as.numeric']
	para = R.FloatVector((success_arm1, failure_arm1, success_arm2, failure_arm2, tspostdiff_thresh))
	assignment = ts_postdiff(para)
	assignment = as_numeric(assignment)
	arm, is_ur = tuple(assignment)

	if arm == 1:
		# Arm 1 chosen
		version_to_show = arm1_success_failure[0]
	else:
		# Arm 2 chosen
		version_to_show = arm2_success_failure[0]

	#log whether was chosen by ts or ur
	ur_or_ts, created = Variable.objects.get_or_create(name="UR_or_TS")
	if is_ur == 1:
		Value.objects.create(
      		variable=ur_or_ts, value=0.0, text="UR", 
            learner=context["learner"], mooclet=context["mooclet"],
			version=version_to_show
   		)
		version_dict = model_to_dict(version_to_show)
		version_dict["selection_method"] = "uniform_random"
		return version_dict
	else:
		#log policy chosen
		Value.objects.create(
      		variable=ur_or_ts, value=1.0, text="TS", 
            learner=context["learner"], mooclet=context["mooclet"],
			version=version_to_show
		)
		version_dict = model_to_dict(version_to_show)
		version_dict["selection_method"] = "thompson_sampling_postdiff"
		return version_dict


def ts_sample_with_r(versions_dict, context):
	"""
	Inputs are a threshold and a dict of versions to successes and failures e.g.:
	{version1: {successess: 1, failures: 1}.
	version2: {successess: 1, failures: 1}, ...}
	"""
	import rpy2.robjects as R

	version_success_failure = list(versions_dict.items())
	arm1_success_failure, arm2_success_failure = version_success_failure[0], version_success_failure[1]
	success_arm1, failure_arm1 = arm1_success_failure[1]["successes"], arm1_success_failure[1]["failures"]
	success_arm2, failure_arm2 = arm2_success_failure[1]["successes"], arm2_success_failure[1]["failures"]

	ts = R.r(
		r'''
		function(para){
			sa <- para[1]
			fa <- para[2]
			sb <- para[3]
			fb <- para[4]
			draw_a <- rbeta(1, sa, fa)
			draw_b <- rbeta(1, sb, fb)
			
			arm <- 1
			if(draw_a < draw_b) {
				arm <- 2
			}
			return(arm)
		}
		'''
    )

	as_numeric = R.r['as.numeric']
	para = R.FloatVector((success_arm1, failure_arm1, success_arm2, failure_arm2))
	assignment = ts(para)
	assignment = as_numeric(assignment)
	arm = assignment[0]

	if arm == 1:
		# Arm 1 chosen
		version_to_show = arm1_success_failure[0]
	else:
		# Arm 2 chosen
		version_to_show = arm2_success_failure[0]

	version_dict = model_to_dict(version_to_show)
	version_dict["selection_method"] = "thompson_sampling"
	return version_dict
