from django.urls import path
from .views import (
    DocumentUploadView, DocumentListView, DocumentDetailView,
    ComplianceInsightListView, RegulatoryFrameworkListView, RequirementListView
)

urlpatterns = [
    path('upload-and-analyze/', DocumentUploadView.as_view(), name='upload-and-analyze'),
    path('documents/', DocumentListView.as_view(), name='document-list'),
    path('documents/<str:doc_id>/', DocumentDetailView.as_view(), name='document-detail'),
    path('compliance-insights/', ComplianceInsightListView.as_view(), name='compliance-insight-list'),
    path('frameworks/', RegulatoryFrameworkListView.as_view(), name='framework-list'),
    path('requirements/', RequirementListView.as_view(), name='requirement-list'),
]