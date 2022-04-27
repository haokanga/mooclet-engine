from rest_framework import viewsets
from rest_pandas import PandasView
from .models import *
from .serializers import *
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from django.db.models import Sum
from django.shortcuts import get_object_or_404
import pandas as pd
import numpy as np
import json

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


# class getBinaryContextualImputer(APIView):
#     def get(self, request):
#         req = json.loads(request.body)

#         if req['name'] is None or req['mooclet'] is None:
#             return Response({'error':'invalid'}, status=500)

#         imputer = {}
#         values = Value.objects.filter(variable__name=req['name'], mooclet=req['mooclet'])
#         num_values = Value.objects.filter(variable__name=req['name'], mooclet=req['mooclet']).count()

#         if num_values == 0:
#             imputer['imputer'] = np.random.choice([0, 1], 1, p=[0.5, 0.5])
#         else:
#             sum_values = values.aggregate(Sum('value'))
#             sum_values = sum_values['value__sum']
#             binary_t_prop = sum_values/num_values
#             imputer['imputer'] = np.random.choice([0, 1], 1, p=[1-binary_t_prop, binary_t_prop])

#         return Response(imputer)


class ContextualImputer(APIView):
    def post(self, request):
        req = json.loads(request.body)

        if not all(key in req for key in ("learner", "mooclet", "policy")):
            return Response({"error": "invalid request"}, status=500)

        mooclet = Mooclet.objects.get(pk=req["mooclet"])
        learner = Learner.objects.get(name=req["learner"])
        policy = Policy.objects.get(pk=req["policy"])

        mooclet_params = PolicyParameters.objects.get(mooclet=mooclet, policy=policy)
        parameters = mooclet_params.parameters
        if "contexts" in req:
            contextual_vars = req["contexts"]
        else:
            contextual_vars = list(filter(lambda context: context != "version", parameters["contextual_variables"]))

        imputer = {}
        for context_var in contextual_vars:
            if context_var not in parameters["contextual_variables"]:
                return Response({"error": f"contextual variable {context_var} is invalid"}, status=500)
            variable = Variable.objects.filter(name=context_var).last()
            val_type = variable.value_type
            print(f"variable type: {val_type}")
            val_min = variable.min_value
            val_max = variable.max_value
            sample_thres = variable.sample_thres
            values = Value.objects.filter(variable__name=context_var, mooclet=mooclet)
            num_values = values.count()

            if num_values == 0 or num_values < sample_thres:
                if val_type != "continuous":
                    sample = np.random.choice(np.arange(val_min, val_max + 1))
                else:
                    sample = np.random.uniform(val_min, val_max)
                sample = (sample - val_min) / (val_max - val_min)
            else:
                val_lst = list(values.value_list("value"))
                sample = np.random.choice(val_lst)
            
            Value.objects.create(
                variable=variable, 
                value=sample, 
                text="Init Context",
                learner=learner,
                mooclet=mooclet
            )

            imputer[context_var] = sample

        return Response({"imputers": imputer})
