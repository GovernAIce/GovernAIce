
# Register your models here.
from django.contrib import admin
from .models import Document, FrameworkSelection, ComplianceInsight, RegulatoryFramework, Requirement

admin.site.register(Document)
admin.site.register(FrameworkSelection)
admin.site.register(ComplianceInsight)
admin.site.register(RegulatoryFramework)
admin.site.register(Requirement)