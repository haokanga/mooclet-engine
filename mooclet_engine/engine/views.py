from rest_framework import viewsets
from rest_pandas import PandasView
from .models import *
from .serializers import *
from .utils.data_downloader_utils import set_if_not_none, map_version_to_reward
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from django.http import HttpResponse
from io import BytesIO
from django.db.models import Sum, Q
from django.shortcuts import get_object_or_404
import pandas as pd
import numpy as np
import json
import csv

# rest framework viewsets

class MoocletViewSet(viewsets.ModelViewSet):


    queryset = Mooclet.objects.all()
    serializer_class = MoocletSerializer
    #lookup_field = 'name'
    search_fields = ('name',)

    @action(detail=True)
    def test(self, request, pk=None):
        return Response({'test':'hi'})

    @action(detail=True)
    def run(self, request, pk=None):
        policy = request.GET.get('policy',None)
        context = {}
        learner = None
        if request.GET.get('user_id', None):
            learner, created = Learner.objects.get_or_create(name=request.GET.get('user_id', None))
        elif request.GET.get('learner', None):
            learner, created = Learner.objects.get_or_create(name=request.GET.get('learner', None))
        context['learner'] = learner
        version = self.get_object().run(context=context)
        #print version
        Version, created = Variable.objects.get_or_create(name='version')
        #TODO: clean this up
        if type(version) is dict:
            version_id = version['id']
            version_name = version['name']
            serialized_version = version
        else:
            version_id = version.id
            version_name = version.name
            serialized_version = VersionSerializer(version).data#.save()
            #serialized_version = serialized_version.data

        version_shown = Value(
                            learner=learner,
                            variable=Version,
                            mooclet=self.get_object(),
                            policy=self.get_object().policy,
                            version_id=version_id,
                            value=version_id,
                            text=version_name
                            )
        if "policy_id" in serialized_version:
            version_shown.policy = Policy.objects.get(id=serialized_version["policy_id"])
        version_shown.save()
        return Response(serialized_version)

class VersionViewSet(viewsets.ModelViewSet):
    queryset = Version.objects.all()
    #lookup_field = 'name'
    multiple_lookup_fields = ('name', 'id')
    serializer_class = VersionSerializer
    filter_fields = ('mooclet', 'mooclet__name',)
    search_fields = ('name', 'mooclet__name',)

    # def get_object(self):
    #     queryset = self.get_queryset()
    #     filter = {}
    #     for field in self.multiple_lookup_fields:
    #         try:
    #             filter[field] = self.kwargs[field]
    #         except:
    #             pass

    #     obj = get_object_or_404(queryset, **filter)
    #     self.check_object_permissions(self.request, obj)
    #     return obj

class VersionNameViewSet(viewsets.ModelViewSet):
    queryset = Version.objects.all()
    #lookup_field = 'name'
    multiple_lookup_fields = ('name', 'id')
    serializer_class = VersionSerializer
    filter_fields = ('mooclet', 'mooclet__name',)
    search_fields = ('name', 'mooclet__name',)

class VariableViewSet(viewsets.ModelViewSet):
    queryset = Variable.objects.all()
    #lookup_field = 'name'
    serializer_class = VariableSerializer
    search_fields = ('name',)

class ValueViewSet(viewsets.ModelViewSet):
    queryset = Value.objects.all()
    serializer_class = ValueSerializer
    filter_fields = ('learner', 'variable', 'learner__name', 'variable__name', 'mooclet', 'mooclet__name', 'version', 'version__name', 'policy',)
    search_fields = ('learner__name', 'variable__name',)
    ordering_fields = ('timestamp','learner', 'variable', 'learner__name', 'variable__name', 'mooclet', 'mooclet__name', 'version', 'version__name',)

    @action(detail=False, methods=['POST'])
    def create_many(self, request, pk=None):
        queryset = Value.objects.all()
        serializer = ValueSerializer(many=True, data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data, status=201)
        else:
            return Response({'error':'invalid'}, status=500)

    @action(detail=False, methods=['POST'])
    def create_many_fromobj(self, request, pk=None):
        queryset = Value.objects.all()
        print("Data:")
        print(request.data)

        vals = request.data[list(request.data.keys())[0]]
        try:
            vals = json.loads(vals)
        except:
            pass
        serializer = ValueSerializer(many=True, data=vals)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data, status=201)
        else:
            return Response({'error':'invalid'}, status=500)

class PolicyViewSet(viewsets.ModelViewSet):
    queryset = Policy.objects.all()
    serializer_class = PolicySerializer
    #lookup_field = 'name'
    search_fields = ('name',)

class LearnerViewSet(viewsets.ModelViewSet):
    queryset = Learner.objects.all()
    #lookup_field = 'name'
    serializer_class = LearnerSerializer
    search_fields = ('name','environment')
    filter_fields = ('name','environment')

class PandasValueViewSet(PandasView):
    queryset = Value.objects.all()
    serializer_class = ValueSerializer
    filter_fields = ('learner', 'variable', 'learner__name', 'variable__name', 'mooclet', 'mooclet__name', 'version', 'version__name',)
    search_fields = ('learner__name', 'variable__name',)


class PandasLearnerValueViewSet(PandasView):
    queryset = Value.objects.all()
    serializer_class = ValueSerializer
    filter_fields = ('learner', 'variable', 'learner__name', 'variable__name', 'mooclet', 'mooclet__name', 'version', 'version__name',)
    search_fields = ('learner__name', 'variable__name',)

    def transform_dataframe(self, dataframe):
        data = dataframe
        data1= data.pivot_table(index='id', columns='variable')['value']
        data = pd.concat([data,data1],axis=1).set_index('learner')
        del data['variable'],data['value']#,data['index']
        list_ = data.columns
        data_transformed = data.groupby(level=0).apply(lambda x: x.values.ravel()).apply(pd.Series)

        for f in data_transformed.columns:
            data_transformed=data_transformed.rename(columns={f:list_[int(f)%len(list_)]+'_a'+str(int(f/len(list_))+1)})
            #dataframe.some_pivot_function(in_place=True)
        return data_transformed

# class EnvironmentViewSet(viewsets.ModelViewSet):
#     queryset = Environment.objects.all()
#     serializer_class = EnvironmentSerializer

# class UserViewSet(viewsets.ModelViewSet):
#     queryset = User.objects.all()
#     serializer_class = UserSerializer

class PolicyParametersViewSet(viewsets.ModelViewSet):
    queryset = PolicyParameters.objects.all()
    serializer_class = PolicyParametersSerializer
    filter_fields = ('mooclet', 'policy')

class PolicyParametersHistoryViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = PolicyParametersHistory.objects.all()
    serializer_class = PolicyParametersHistorySerializer
    filter_fields = ('mooclet', 'policy')

class ContextualImputer(APIView):
    def post(self, request):
        req = json.loads(request.body)

        if not all(key in req for key in ("learner", "mooclet", "policy")):
            return Response({"error": "invalid request"}, status=500)

        try:
            mooclet = Mooclet.objects.get(pk=req["mooclet"])
        except:
            return Response({"error": "mooclet not found"}, status=404)

        try:
            learner = Learner.objects.get(name=req["learner"])
        except:
            return Response({"error": "learner not found"}, status=404)

        try:
            policy = Policy.objects.get(pk=req["policy"])
        except:
            return Response({"error": "policy not found"}, status=404)

        try:
            mooclet_params = PolicyParameters.objects.get(mooclet=mooclet, policy=policy)
        except:
            return Response({"error": "policy parameters not found in mooclet"}, status=404)

        parameters = mooclet_params.parameters
        if "contexts" in req:
            contextual_vars = req["contexts"]
        else:
            contextual_vars = list(filter(lambda context: context != "version", parameters["contextual_variables"]))

        imputer = {}
        context_samples = []
        for context_var in contextual_vars:
            if context_var not in parameters["contextual_variables"]:
                return Response({"error": f"contextual variable {context_var} is invalid"}, status=500)
            variable = Variable.objects.filter(name=context_var).last()
            val_type = variable.value_type
            val_min = variable.min_value
            val_max = variable.max_value
            sample_thres = variable.sample_thres
            values = Value.objects.filter(variable__name=context_var, mooclet=mooclet)
            num_values = values.count()

            if num_values == 0 or num_values < sample_thres:
                if val_type != Variable.CONTINUOUS:
                    sample = np.random.choice(np.arange(val_min, val_max + 1))
                else:
                    sample = np.random.uniform(val_min, val_max)
                sample = (sample - val_min) / (val_max - val_min)
            else:
                val_lst = list(float(val_tup[0]) for val_tup in values.values_list("value"))
                sample = np.random.choice(val_lst)

            context_samples.append((variable, sample))
            imputer[context_var] = sample

        for context in context_samples:
            variable, sample = context
            Value.objects.create(
                variable=variable,
                value=sample,
                text="Init Context",
                learner=learner,
                mooclet=mooclet
            )

        return Response({"imputers": imputer})


class ExportExcelValues(APIView):
    VARIABLE_NAMES = {
        "contextual": {
            "aliases": [
                "contextual_variables"
            ]
        },
        "reward": {
            "aliases": [
                "outcome_variable",
                "outcome_variable_name"
            ]
        }
    }

    # Url: /datadownload/?mooclet=XX/
    def get(self, request):
        # Find a Model instance of Mooclet by mooclet id or name.
        mooclet_arg_dict = {}
        set_if_not_none(mooclet_arg_dict, "pk", request.query_params["mooclet"])
        set_if_not_none(mooclet_arg_dict, "name", request.query_params["mooclet__name"])
        if len(mooclet_arg_dict) == 0:
            return Response({"error": "invalid request"}, status=500)

        try:
            mooclet = Mooclet.objects.get(**mooclet_arg_dict)
        except:
            return Response({"error": "Mooclet not found"}, status=404)
        
        # Find a QuerySet instance of Version by version id or name. 
        # This instance is optional. If no arguments are given, then look for 
        # all version instance for the given mooclet.
        version_arg_dict = {"mooclet": mooclet}
        set_if_not_none(version_arg_dict, "pk", request.query_params["version"])
        set_if_not_none(version_arg_dict, "name", request.query_params["version__name"])
        
        try:
            versions = Version.objects.filter(**version_arg_dict)
        except:
            return Response({"error": "Version not found"}, status=404)
        
        # Find a QuerySet instance of Learner by learner id or name. 
        # This instance is optional. If no arguments are given, then look for 
        # all learner instances for the given mooclet.
        learner_arg_dict = {}
        set_if_not_none(learner_arg_dict, "pk", request.query_params["learner"])
        set_if_not_none(learner_arg_dict, "name", request.query_params["learner__name"])
        
        try:
            learners = Learner.objects.filter(**learner_arg_dict)
        except:
            return Response({"error": "Learner not found"}, status=404)

        # Find a QuerySet instance for Variable by variable id or name. 
        # This instance is optional. If no arguments are given, then look for 
        # all variable instances for the given mooclet.

        # TODO: If user just give one veriable and ask for datadownloading, the csv file can be very different
        #   from the default one. For example, we can only track such variable in the Value QuerySet and we 
        #   don't have any information about whether this is a reward or a context, or something else.
        #   One prototype: If user ONLY ask for one variable, then we give them a csv file of such variable. 
        #       Which means we don't need to map versions to such variable.
        variable_arg_dict = {}
        set_if_not_none(variable_arg_dict, "pk", request.query_params["variable"])
        set_if_not_none(variable_arg_dict, "name", request.query_params["variable__name"])
        
        try:
            variables = Variable.objects.filter(**variable_arg_dict)
        except:
            return Response({"error": "Variable not found"}, status=404)
        
        # Find a QuerySet instance of Policy by policy id. 
        # This instance is optional. If no arguments are given, then look for 
        # all policy instances for the given mooclet.
        policy_arg_dict = {}
        set_if_not_none(policy_arg_dict, "pk", request.query_params["policy"])
        
        try:
            policies = Policy.objects.filter(**policy_arg_dict)
        except:
            return Response({"error": "Policy not found"}, status=404)

        # The Model Queries
        # 1) select_parameters - Policy parameters
        if len(policy_arg_dict) == 0:
            # Find a QuerySet of PolicyParameters instances by mooclet.
            select_parameters = PolicyParameters.objects.filter(mooclet=mooclet)

            # # Check if contextual policy is used in the mooclet instance.
            # contextual_parameters = select_parameters.filter(policy__in=Policy.objects.filter(name__contains="contextual"))
            
            # If no policy specified, Update policies QuerySet so that only contains policies 
            # related to the mooclet instance.
            try:
                policies = Value.objects.filter(mooclet=mooclet).exclude(policy__isnull=True).only('policy').distinct()
            except:
                return Response({"error": "Unknown field: 'policy'"}, status=400)
        else:
            # Find a QuerySet of PolicyParameters instances by mooclet and policy.
            select_parameters_kargs = {
                "mooclet": mooclet,
                "policy": policies.first()
            }
            select_parameters = PolicyParameters.objects.filter(**select_parameters_kargs)
        
        if not select_parameters.exists():
            return Response({"error": "Policy parameter not found"}, status=404)
        
        # Get a set of all variables and reward variables related to the mooclet instance.
        all_variables = {}
        reward_variables = {}
        try:
            for param in select_parameters:
                field = param._meta.get_field(parameters)
                parameters = dict(param.parameters)

                # Add all contextual variables.
                for contextual_param_alias in self.VARIABLE_NAMES["contextual"]["aliases"]:
                    if contextual_param_alias in parameters:
                        all_variables.union(set(parameters[contextual_param_alias]))
                
                # Add all reward variables.
                for reward_param_alias in self.VARIABLE_NAMES["reward"]["aliases"]:
                    if reward_param_alias in parameters:
                        all_variables.union(set([parameters[reward_param_alias]]))
                        reward_variables.union(set([parameters[reward_param_alias]]))
        except FieldDoesNotExist:
            return Response({"error": "Unknown field: 'parameters'"}, status=400)

        # If no variable specified, update variables QuerySet so that only contains variables 
        # related to the mooclet instance.
        if len(variable_arg_dict) == 0:
            variables = variables.filter(name__in=list(all_variables))
        reward_variables = variables.filter(name__in=list(reward_variables))

        # 2) select_param_histories - Policy parameters history
        # This QuerySet is sorted by creation_time (oldest to newest).
        # This QuerySet can specified by policy.
        select_param_histories = PolicyParametersHistory.objects.order_by("creation_time").filter(Q(mooclet=mooclet) & Q(policy__in=policies))

        # 3) select_param_histories - Value
        # This QuerySet is sorted by timestamp (oldest to newest).
        # This QuerySet can specified by version, learner, variable, or policy.

        # mooclet, learner, and variables cannot be NULL.
        # Exclude all values which has NULL in mooclet fields.
        select_values_kargs = {}
        set_if_not_none(select_values_kargs, "mooclet", mooclet)
        if len(learner_arg_dict) != 0:
            set_if_not_none(select_values_kargs, "learner", learners.first())
        if len(variable_arg_dict) != 0:
            set_if_not_none(select_values_kargs, "variable", variables.first())
        
        # Find a QuerySet object of Value by mooclet, learner, and variable.
        select_values = Value.objects.order_by("timestamp").filter(**select_values_kargs)

        # version and policy can be NULL. 
        # Need to filter QuerySet value by version and policy but allow these two fields to be NULL.
        if len(version_arg_dict) != 0:
            select_values = select_values.filter(Q(version__isnull=True) | Q(version__in=versions))
        if len(policy_arg_dict) != 0:
            select_values = select_values.filter(Q(policy__isnull=True) | Q(policy__in=policies))
        
        # Exclude all values which has NULL in learner and variable fields.
        select_values = select_values.exclude(Q(learner__isnull=True) | Q(variable__isnull=True))

        # Check if there are any existing values.
        if not select_values.exists():
            return Response({"error": "Value not found"}, status=404)

        # TODO: Requesting data for a specific variable (reward or context, or something else).

        with BytesIO() as b:
            with pd.ExcelWriter(b, engine='xlsxwriter') as writer:
                # Generate each data csv file for each policy
                for policy_idx, policy in enumerate(policies):
                    datapoint_frames = []

                    single_param_histories = select_param_histories.filter(policy=policy)
                    single_parameters = select_parameters.filter(policy=policy)

                    prev_checkpoint = None
                    for update_count, param_history in enumerate(single_param_histories):
                        curr_checkpoint = param_history.creation_time

                        # Slice Value QuerySet by the creation_time of current checkpoint 
                        # and previous checkpoint (if exists).
                        if prev_checkpoint:
                            batch_values = select_values.filter(
                                Q(timestamp__gte=prev_checkpoint) & Q(timestamp__lt=curr_checkpoint)
                            )
                        else:
                            batch_values = select_values.filter(Q(timestamp__lt=curr_checkpoint))

                        # QuerySet batch_values is empty means no new values for updating parameters.
                        if not batch_values.exists():
                            # Slice the last part of the batch values
                            batch_values = select_values.filter(Q(timestamp__gte=curr_checkpoint))

                            if not batch_values.exists():
                                break
                            
                            data = map_version_to_reward(
                                batch_values, 
                                mooclet, 
                                policy, 
                                reward_variables,
                                variables,
                                update_group=update_count,
                                policy_params=single_parameters
                            )
                        else:
                            data = map_version_to_reward(
                                batch_values, 
                                mooclet, 
                                policy, 
                                reward_variables,
                                variables,
                                update_group=update_count,
                                policy_params_history=param_history
                            )

                        datapoint_frames.append(data)
                        prev_checkpoint = curr_checkpoint
                    
                    policy_datapoints = pd.concat(datapoint_frames)

                    # ONLY export excel file if dataframe is not empty.
                    if len(policy_datapoints):
                        policy_datapoints.to_excel(writer, sheet_name="{}_{}".format(policy.name, policy_idx))

            filename = "{}.xlsx".format(mooclet.name.replace(" ", "_"))
            # imported from django.http
            res = HttpResponse(
                b.getvalue(), # Gives the Byte string of the Byte Buffer object
                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            res['Content-Disposition'] = f'attachment; filename={filename}'
            return res
