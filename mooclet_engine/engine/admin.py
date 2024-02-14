from django.contrib import admin
from .models import *

# Register your models here.


admin.site.register(Environment)
admin.site.register(Mooclet)
admin.site.register(Version)
admin.site.register(Variable)
# admin.site.register(Value)
admin.site.register(Policy)
# admin.site.register(Learner)
# admin.site.register(PolicyParameters)

@admin.register(PolicyParametersHistory)
class PolicyParameterHistoryAdmin(admin.ModelAdmin):
    readonly_fields = ('creation_time',)

@admin.register(Value)
class ValueAdmin(admin.ModelAdmin):
    list_display = ('variable', 'learner', 'mooclet', 'value', 'timestamp')
    search_fields = ('value', 'text', 'variable__name',)
    list_filter = ('timestamp',)

@admin.register(Learner)
class LearnerAdmin(admin.ModelAdmin):
    list_display = ('name', 'environment', 'learner_id')
    search_fields = ('name', 'learner_id')
    list_filter = ('environment',)

@admin.register(PolicyParameters)
class PolicyParametersAdmin(admin.ModelAdmin):
    list_display = ('mooclet', 'policy', 'latest_update')
    search_fields = ('parameters',)
    list_filter = ('latest_update', 'policy', 'mooclet',)
