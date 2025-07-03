from rest_framework import serializers
from .models import *
from django.core.validators import MinLengthValidator

class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['doc_id', 'title', 'upload_date', 'risk_classification']

class FrameworkSelectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = FrameworkSelection
        fields = ['framework', 'analysis_status']

class ComplianceInsightSerializer(serializers.ModelSerializer):
    related_requirements = serializers.SlugRelatedField(many=True, read_only=True, slug_field='requirement_id')
    framework = serializers.SlugRelatedField(slug_field='name', read_only=True)

    class Meta:
        model = ComplianceInsight
        fields = ['framework', 'insight', 'created_at', 'related_requirements']

class RegulatoryFrameworkSerializer(serializers.ModelSerializer):
    class Meta:
        model = RegulatoryFramework
        fields = ['name', 'description', 'version']

class RequirementSerializer(serializers.ModelSerializer):
    framework = serializers.SlugRelatedField(slug_field='name', read_only=True)

    class Meta:
        model = Requirement
        fields = ['requirement_id', 'framework', 'description', 'category']

class DocumentUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    frameworks = serializers.ListField(
        child=serializers.ChoiceField(choices=[
            ('EU_AI_ACT', 'EU AI Act'),
            ('NIST_RMF', 'NIST AI RMF'),
            ('CPRA', 'CPRA'),
        ])
    )

class DocumentDetailSerializer(serializers.ModelSerializer):
    framework_selections = FrameworkSelectionSerializer(many=True, read_only=True)
    compliance_insights = ComplianceInsightSerializer(many=True, read_only=True)

    class Meta:
        model = Document
        fields = ['doc_id', 'title', 'upload_date', 'risk_classification', 'framework_selections', 'compliance_insights']

class AnalysisResultSerializer(serializers.Serializer):
    doc_id = serializers.CharField()
    filename = serializers.CharField()
    insights = serializers.ListField(child=serializers.DictField())


class ProductInfoUploadSerializer(serializers.Serializer):
    product_info = serializers.CharField(
        validators=[MinLengthValidator(10)],
        help_text="Enter the product details in the textbox."
    )
    frameworks = serializers.ListField(
        child=serializers.ChoiceField(choices=[
            ('EU_AI_ACT', 'EU AI Act'),
            ('NIST_RMF', 'NIST AI RMF'),
            ('CPRA', 'CPRA'),
        ]),
        help_text="List of regulatory frameworks to analyze against (e.g., ['EU_AI_ACT', 'NIST_RMF', 'CPRA'])."
    )