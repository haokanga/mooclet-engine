from __future__ import unicode_literals
from django.contrib.postgres.fields import JSONField
from django.db import models
from django.urls import reverse
from django.shortcuts import redirect
from . import policies
from django.http import Http404
from django.utils import timezone

from numpy.random import choice

class Environment(models.Model):
    name = models.CharField(max_length=200,default='')

    def __unicode__(self):
        return "{} {}: {}".format(self.__class__.__name__, self.pk, self.name)


class Mooclet(models.Model):
    name = models.CharField(max_length=100,default='', unique=True)
    policy = models.ForeignKey('Policy',blank=True,null=True, on_delete=models.SET_NULL)
    environment = models.ForeignKey(Environment, blank=True, null=True, default=None, on_delete=models.SET_NULL)
    mooclet_id = models.PositiveIntegerField(blank=True,null=True)

    # class Meta:
    #     unique_together = ('environment','mooclet_id')

    def __str__(self):
        return "{}: {}".format(self.__class__.__name__, self.name)

    def run(self, policy=None, context={}):
        context['mooclet'] = self
        if not self.version_set.exists():
            raise Http404('mooclet has no versions')
        if not policy:
            if self.policy:
                policy = self.policy
            else:
                # print 'no policy found'
                raise Http404('no policy found')

        version = policy.run_policy(context)

        return version


class Version(models.Model):
    '''
    Mooclet version
    '''
    
    name = models.CharField(max_length=200,default='')
    mooclet = models.ForeignKey(Mooclet,on_delete=models.SET_NULL,null=True)
    text = models.TextField(blank=True,default='')
    version_id = models.PositiveIntegerField(blank=True,null=True)
    # mooclet_version_id = models.PositiveIntegerField(blank=True)
    version_json = JSONField(blank=True, null=True)

    # @property
    # def environment(self):
    #     return self.mooclet.environment.pk

    class Meta:
        unique_together = ('mooclet','name')

    def __str__(self):
        return "{} {}: {}".format(self.__class__.__name__, self.pk, self.name)


# class MoocletPolicyState(models.Model):
# 	pass


class Learner(models.Model):
    name = models.CharField(max_length=10000,unique=True)
    environment = models.ForeignKey(Environment,blank=True,null=True, default=None, on_delete=models.SET_NULL)
    learner_id = models.PositiveIntegerField(blank=True,null=True)

    # class Meta:
        #unique_together = ('environment','learner_id')


class Variable(models.Model):
    BINARY = 'BIN'
    ORDINARY = 'ORD'
    CONTINUOUS = 'CONT'
    VARIABLE_TYPE = (
        (BINARY, 'binary'), 
        (ORDINARY, 'ordinary'), 
        (CONTINUOUS, 'continuous')
    )
    name = models.CharField(max_length=100, unique=True)
    environment = models.ForeignKey(Environment,blank=True,null=True, default=None, on_delete=models.SET_NULL)
    variable_id = models.PositiveIntegerField(blank=True,null=True)
    min_value = models.FloatField(default=0.0)
    max_value = models.FloatField(default=1.0)
    value_type = models.CharField(max_length=100, choices=VARIABLE_TYPE, default=BINARY)
    sample_thres = models.PositiveIntegerField(default=15)

    def __str__(self):
        return self.name
        #return "{} {}: {}".format(self.__class__.__name__, self.pk, self.name)

    def get_data(self,context=None):
        '''
        return relevant value objects for the variable type
        '''
        #assume if there is a context, it always contains a mooclet(?)
        if context:
            if 'mooclet' in context:
                values = self.value_set.filter(version__in=context['mooclet'].version_set.all())
                if 'version' in context:
                    values = values.filter(version=context['version'])
                if 'policy' in context:
                    values = values.filter(policy__name=context["policy"])
                return values
        else:
            return self.value_set.all()


    def get_data_dicts(self,context=None):
        '''
        return relevant values for the variable type, as a list of dicts
        '''
        return self.get_data(context).values()


class Value(models.Model):
    '''
    user variable observation, can be associated with either course, mooclet, or mooclet version
    examples of user variables:
        course-level: general student characteristics
        quiz-level: number of attempts
        mooclet: ?
        version: student rating of an explanation, instructors prior judgement
    '''
    variable = models.ForeignKey(Variable,on_delete=models.SET_NULL,null=True)

    learner = models.ForeignKey(Learner,null=True,blank=True, on_delete=models.SET_NULL)
    mooclet = models.ForeignKey(Mooclet,null=True,blank=True, on_delete=models.SET_NULL)
    version = models.ForeignKey(Version,null=True,blank=True, on_delete=models.SET_NULL)
    policy = models.ForeignKey('Policy',null=True,blank=True, on_delete=models.SET_NULL)

    value = models.FloatField(blank=True,null=True)
    text = models.TextField(blank=True,default='')
    timestamp = models.DateTimeField(null=True,default=timezone.now)

    class Meta:
        indexes = [
                    models.Index(fields=['mooclet', 'learner', 'version'], name='value_primary_idx'),
                    models.Index(fields=['mooclet'], name='value_mooclet_idx'),
                    models.Index(fields=['learner'], name='value_learner_idx'),
                    models.Index(fields=['learner','timestamp'], name='value_timestamp_idx')
        ]
        ordering = ['learner', '-timestamp']

    # value_id = models.PositiveIntegerField(blank=True)

    # class Meta:
    #     unique_together = ('environment','value_id')

    # @property
    # def environment(self):
    #     return self.variable.environment


class Policy(models.Model):
    name = models.CharField(max_length=100)
    environment = models.ForeignKey(Environment,null=True,blank=True,default=None, on_delete=models.SET_NULL)
    policy_id = models.PositiveIntegerField(blank=True)
    # variables = models.ManyToManyField('Variable') # might use this for persistent "state variables"?

    class Meta:
        verbose_name_plural = 'policies'
        unique_together = ('environment','policy_id')

    def __str__(self):
        return self.name

    def get_policy_function(self):
        try:
            return getattr(policies, self.name)
        except:
            print ("policy function matching specified name not found")
            # TODO look through custom user-provided functions
            return None

    def get_variables(self):
        # TODO implement returning all, subsets, etc.
        # return self.variables.all()
        return Variable.objects.all()

    def run_policy(self, context):
        # insert all version ids here?
        policy_function = self.get_policy_function()
        policy_parameters = None

        try:
            policy_parameters = PolicyParameters.objects.get(mooclet=context['mooclet'], policy=self)
            # print "params"
            # print policy_parameters
        except:
            pass
        context['policy_parameters'] = policy_parameters
        variables = self.get_variables()
        #variables = []
        version = policy_function(variables,context)
        
        forbid_policy_names = [
            "choose_mooclet_group",
            "choose_policy_group"
        ]
        
        if self.name not in forbid_policy_names and 'allowed_versions' in context:
            if type(version) is dict:
                version_id = version["id"]
            else:
                version_id = version.id
            
            allowed_versions = context['allowed_versions']
            max_time = context.get("maximum_allowed", 20)
            
            print(f"version_id: {version_id}")
            print(f"allowed_versions: {allowed_versions}")
            print(f"max_time: {max_time}")
            
            count = 1
            
            print(f"{'-' * 12} COUNT {count} {'-' * 12}")
            print("filter: {}".format(allowed_versions.filter(pk=version_id).exists()))
            
            while (not allowed_versions.filter(pk=version_id).exists() and count < max_time):
                version = policy_function(variables,context)
                if type(version) is dict:
                    version_id = version["id"]
                else:
                    version_id = version.id
                count += 1
                
                print(f"version_id: {version_id}")
                
                print(f"{'-' * 12} COUNT {count} {'-' * 12}")
            
            if count == max_time:
                all_allowed_pks = allowed_versions.values_list('pk', flat=True)
                version_pk = choice(all_allowed_pks)
                version = allowed_versions.get(pk=version_pk)
                
                print(f"tried {max_time} times")
                print(f"randomly select an arm: {version}")
        #version = policies.uniform_random(variables, context)

        return version

class PolicyParameters(models.Model):
    mooclet = models.ForeignKey(Mooclet, null=True, blank=True, default=None,on_delete=models.SET_NULL)
    policy = models.ForeignKey(Policy,on_delete=models.SET_NULL,null=True)
    #make this a jsonfield
    parameters = JSONField(null=True, blank=True)
    latest_update = models.DateTimeField(null=True, blank=True)
    #model = JSONField()
    class Meta:
        verbose_name_plural = 'policyparameters'
        unique_together = ('mooclet', 'policy')

    def __str__(self):
        return "{}, Policy: {}".format(self.mooclet, self.policy)

class PolicyParametersHistory(models.Model):
    mooclet = models.ForeignKey(Mooclet, null=True, blank=True, default=None,on_delete=models.SET_NULL)
    policy = models.ForeignKey(Policy,on_delete=models.SET_NULL, null=True)
    #make this a jsonfield
    parameters = JSONField(null=True, blank=True)
    creation_time = models.DateTimeField(null=True, blank=True, auto_now_add=True)
    #model = JSONField()
    class Meta:
        verbose_name_plural = 'policyparameterhistories'
        ordering = ['creation_time']
        #unique_together = ()

    def __str__(self):
        return "{}, Policy: {}, created: {}".format(self.mooclet, self.policy, self.creation_time)

    @classmethod
    def create_from_params(cls, params):
        param_history = cls(mooclet=params.mooclet, 
                                policy=params.policy,
                                parameters=params.parameters,
                                )
        param_history.save()
        return param_history


