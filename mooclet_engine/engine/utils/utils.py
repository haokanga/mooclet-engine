from __future__ import unicode_literals
from numpy.random import choice
from collections import Counter
import pandas as pd
import numpy as np
from django.db.models import Q
from django.apps import apps
import string

def sample_no_replacement(full_set, previous_set=None):
	# print "starting sample_no_replacement"
	# if previous_set:
	# 	print "prev set"
	# 	print previous_set
	# print full_set

	if set(previous_set) == set(full_set):
		#the user has seen each of the conditions of this vairbales at least once
		cond_counts = Counter(previous_set)
		#return a list of tuples of most common e.g.
		#[('a', 5), ('r', 2), ('b', 2)]
		cond_common_order = cond_counts.most_common()
		# print "cond_common_order"
		# print cond_common_order
		if cond_common_order[0][1] == cond_common_order[-1][1]:
			#all conds are evenly assigned, choose randomly
			cond = cond_common_order[choice(len(cond_common_order))]
			#print cond
			cond = cond[0]
		else:
			#choose the one with least assignment
			#(where same value is ordered arbitrarily)
			cond = cond_common_order[-1][0]
	else:
		#subject hasn't seen all versions yet
		cond_choices = set(full_set) - set(previous_set)
		# print "cond_choices"
		# print cond_choices
		cond = choice(list(cond_choices))

	return cond

def create_design_matrix(input_df, formula, add_intercept = True):
    '''
    :param input_df:
    :param formula: for eaxmple "y ~ x0 + x1 + x2 + x0 * x1 + x1 * x2"
    :param add_intercept: whether to add dummy columns of 1.
    :return: the design matrix as a dataframe, each row corresponds to a data point, and each column is a regressor in regression
    '''

    D_df = pd.DataFrame()
    input_df = input_df.astype(np.float64)

    formula = str(formula)
    # parse formula
    formula = formula.strip()
    all_vars_str = formula.split('~')[1].strip()
    dependent_var = formula.split('~')[0].strip()
    vars_list = all_vars_str.split('+')
    vars_list = list(map(str.strip, vars_list))

    ''''#sanity check to ensure each var used in
    for var in vars_list:
        if var not in input_df.columns:
            raise Exception('variable {} not in the input dataframe'.format((var)))'''

    # build design matrix
    for var in vars_list:
        if '*' in var:
            interacting_vars = var.split('*')
            interacting_vars = list(map(str.strip,interacting_vars))
            D_df[var] = input_df[interacting_vars[0]]
            for i in range(1, len(interacting_vars)):
                D_df[var] *= input_df[interacting_vars[i]]
        else:
            D_df[var] = input_df[var]

    # add dummy column for bias
    if add_intercept:
        D_df.insert(0, 'Intercept', 1.)

    return D_df


def values_to_df(mooclet, policyparams, latest_update=None):
    """
    where variables is a list of variable names
    note: as implemented this will left join on users which can result in NAs
    """
    Value = apps.get_model('engine', 'Value')
    variables = list(policyparams.parameters["contextual_variables"])
    contexts = []
    for context in variables:
        if context != "version":
            contexts.append(context)
    outcome = policyparams.parameters["outcome_variable"]
    action_space = policyparams.parameters["action_space"]
    variables.append(outcome)

    # print("CHECK latest_update:")
    if not latest_update:
        # print("last update is NONE")
        values = Value.objects.filter(variable__name__in=variables, mooclet=mooclet, policy__name="thompson_sampling_contextual").order_by('timestamp')
    else:
        # print("last update is {}".format(latest_update))
        values = Value.objects.filter(variable__name__in=variables, timestamp__gte=latest_update, mooclet=mooclet, policy__name="thompson_sampling_contextual").order_by('timestamp')

    print("VALUES: {}, LENGTH: {}".format(values, len(values)))

    variables.append('user_id')
    # variables.append(outcome)
    variables.remove('version')
    variables.extend(action_space.keys())
    variables.append('version_added_later')
    print("variables: {}".format(variables))
    print("contexts: {}".format(contexts))
    print("outcome: {}".format(outcome))
    vals_to_df = pd.DataFrame({},columns=variables)
    index = 0
    lastest_timestamp = None
    assigned = {}
    for value in values:
        # print("ADDED VALUE: variable: {}, learner: ({}, {}), version: {}, policy: {}, value: {}, text: {}".format(
        #          value.variable.name, value.learner.id, value.learner.name,
        #     value.version.text, value.policy.policy_id, value.value, value.text))
        # skip any values with no learners
        if not value.learner:
            continue

        # print("VARIABLE NAME: {}".format(value.variable.name))

        # skip if value is not version or reward
        if value.variable.name not in ["version", outcome]:
            continue

        if value.variable.name == "version":
            vals_to_df = vals_to_df.append({'user_id': value.learner.id}, ignore_index=True)
            # print("NEW USER!")
            # print(vals_to_df.to_string())
            add_time = value.timestamp
            if not latest_update or add_time >= latest_update:
                vals_to_df.loc[index, 'version_added_later'] = True
            else:
                vals_to_df.loc[index, 'version_added_later'] = False
            # print("version_add_later logged")
            # print(vals_to_df.to_string())
            action_config = policyparams.parameters['action_space']
            # this is the numerical representation from the config
            # IN THIS STEP ALSO GET AN OUTCOME RELATED TO THIS VERSION
            for action in action_config:
                curr_action_config = value.version.version_json[action]
                vals_to_df.loc[index, action] = curr_action_config

            assigned[value.learner.id] = index
            # print("assigned: {}".format(assigned))
            index += 1
            lastest_timestamp = value.timestamp
        else:
            # print("VALUE FOUND: User {}, ".format(value.learner.id))
            # print("assigned: {}".format(assigned))
            if value.learner.id not in assigned:
                continue
            vals_to_df.loc[assigned[value.learner.id], outcome] = value.value
            # print("value logged")
            # print(vals_to_df.to_string())

            # add context value to vals_to_df
            for context in contexts:
                # print("ADD CONTEXTS")
                # find the last context value
                context_values = Value.objects.filter(variable__name=context, mooclet=mooclet,
                                                      learner=value.learner).order_by('-timestamp')
                if lastest_timestamp is not None:
                    context_values = context_values.filter(Q(timestamp__lt=lastest_timestamp)).order_by('-timestamp')
                if context_values.count() > 0:
                    vals_to_df.loc[assigned[value.learner.id], context] = context_values[0].value
                    # print("context logged")
                    # print(vals_to_df.to_string())

    # # curr_user_values = {}
    # # TODO: if the variable is "version" get the mapping to actions
    # for value in values:
    #     print("ADDED VALUE: variable: {}, learner: {}, version: {}, policy: {}, value: {}, text: {}".format(
    #         value.variable, value.learner, value.version, value.policy, value.value, value.text))
    #
    #     # skip any values with no learners
    #     if not value.learner:
    #         continue
    #
    #     # skip if context variables in value list
    #     if value.variable in contexts:
    #         continue
    #
    #     if curr_user is None:
    #         curr_user = value.learner
    #         curr_user_values = {'user_id': curr_user.id}
    #
    #     # append to df if current user is a new user (assume data e.g. reward, link are loaded)
    #     if value.learner.id != curr_user.id:
    #         # add context value to curr_user_values
    #         for context in contexts:
    #             # find the last context value
    #             context_values = Value.objects.filter(variable__name=context, mooclet=mooclet,
    #                                                   learner=curr_user).order_by('-timestamp')
    #             if context_values.count() > 0:
    #                 curr_user_values[context] = context_values[0].value
    #         # append to df
    #         try:
    #             vals_to_df = vals_to_df.append(curr_user_values, ignore_index=True)
    #         except ValueError:
    #             print("duplicate data: {}".format(curr_user_values))
    #             pass
    #         # update current user
    #         curr_user = value.learner
    #         curr_user_values = {'user_id': curr_user.id}
    #
    #     # transform mooclet version shown into dummified action
    #     # todo  silo off version as its own thing??? so that we always get most recent?
    #     if value.variable.name == 'version':
    #             # get timestamp of version values
    #             add_time = value.timestamp
    #
    #             if not latest_update:
    #                 curr_user_values['version_added_later'] = True
    #             elif add_time < latest_update:
    #                 curr_user_values['version_added_later'] = False
    #             else:
    #                 curr_user_values['version_added_later'] = True
    #
    #             action_config = policyparams.parameters['action_space']
    #             # this is the numerical representation from the config
    #             # IN THIS STEP ALSO GET AN OUTCOME RELATED TO THIS VERSION
    #             for action in action_config:
    #                 curr_action_config = value.version.version_json[action]
    #                 curr_user_values[action] = curr_action_config
    #     # UNLESS IT IS AN OUTCOME IN WHICH CASE HANDLE AS ABOVE AND DISCARD
    #     else:
    #         curr_user_values['version_added_later'] = False
    #         curr_user_values[value.variable.name] = value.value
    #
    # # add context value to the last row of dataframe
    # if curr_user:
    #     # add context value to curr_user_values
    #     for context in contexts:
    #         # find the last context value
    #         context_values = Value.objects.filter(variable__name=context, mooclet=mooclet,
    #                                               learner=curr_user).order_by('-timestamp')
    #         if context_values.count() > 0:
    #             curr_user_values[context] = context_values[0].value
    #     try:
    #         vals_to_df = vals_to_df.append(curr_user_values, ignore_index=True)
    #     except ValueError:
    #         print("duplicate data")
    #         print(curr_user_values)
    #         pass

    # print("values df: {}".format(vals_to_df))
    # print("empty? {}".format(vals_to_df.empty))
    if not vals_to_df.empty:
        # print("NOT EMPTY: {}".format(vals_to_df.to_string()))
        assert "version_added_later" in vals_to_df.columns

        vals_to_df = vals_to_df[vals_to_df["version_added_later"] == True]

        if not vals_to_df.empty:
            # print("FILTER ADDED LATER: {}".format(vals_to_df.to_string()))
            vals_to_df = vals_to_df.drop(["version_added_later"], axis=1)
            # print("DROP ADDED LATER: {}".format(vals_to_df.to_string()))
        # else:
        #     print("FILTER ADDED LATER: vals_to_df is EMPTY")

        output_df = vals_to_df.dropna()

        # if not output_df.empty:
        #     print("output_df: {}".format(output_df.to_string()))
        # else:
        #     print("output_df is EMPTY")

        # print(output_df)
        # if not output_df.empty:
        #     print()
        #
        # #drop rows with version added before the latest update
        # if "version_added_later" in output_df:
        #     print(output_df.version_added_later)
        #     output_df = output_df[output_df.version_added_later]
        #     print(output_df)
        #     if not output_df.empty:
        #         output_df.drop(["version_added_later"], axis=1)
        # print(output_df)
    else:
        # print("EMPTY: {}".format(vals_to_df))
        output_df = vals_to_df
    # if vals_to_df :
    #     output_df = pd.concat(vals_to_df)
    #     output_df = output_df.dropna()
    #     #print output_df.head()
    # else:
    #     output_df = pd.DataFrame()
    # print("output_df: {}".format(output_df))

    return output_df
