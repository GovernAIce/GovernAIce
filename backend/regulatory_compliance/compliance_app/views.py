from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, generics
from pymongo import MongoClient
from .models import *
from .serializers import *
from .ingestor import extract_text_from_pdf
from .analyzer import analyze_document
import uuid
import json
from datetime import datetime
from .analyzer import analyze_document
import uuid
import json
from django.core.validators import MinLengthValidator
from rest_framework import serializers

class DocumentUploadView(APIView):
    def post(self, request):
        serializer = DocumentUploadSerializer(data=request.data)
        if serializer.is_valid():
            file = serializer.validated_data['file']
            frameworks = serializer.validated_data['frameworks']
            text = extract_text_from_pdf(file)
            doc_id = str(uuid.uuid4())
            doc = Document.objects.create(doc_id=doc_id, title=file.name)
            mongo_client = MongoClient('mongodb://localhost:27017/')
            mongo_db = mongo_client['regulatory_mongo']
            mongo_db.documents.insert_one({'doc_id': doc_id, 'content': text})
            for framework in frameworks:
                FrameworkSelection.objects.create(document=doc, framework=framework)
            insights = analyze_document(text, frameworks, doc_id)
            for insight in insights:
                framework_obj, _ = RegulatoryFramework.objects.get_or_create(name=insight['framework'])
                insight_obj = ComplianceInsight.objects.create(
                    document=doc,
                    framework=framework_obj,
                    insight=json.dumps(insight)
                )
                if insight.get('key_requirements'):
                    for req in insight['key_requirements']:
                        req_obj, _ = Requirement.objects.get_or_create(
                            framework=framework_obj,
                            requirement_id=f"{framework_obj.name}_{req[:10].replace(' ', '_')}",
                            defaults={'description': req, 'category': 'General'}
                        )
                        insight_obj.related_requirements.add(req_obj)
            # Update FrameworkSelection status to COMPLETED after successful analysis
            for framework in frameworks:
                FrameworkSelection.objects.filter(document=doc, framework=framework).update(analysis_status='COMPLETED')
            for insight in insights:
                if insight['framework'] == 'EU_AI_ACT':
                    doc.risk_classification = insight['risk_classification']
                    doc.save()
                    break
            response_data = {
                'doc_id': doc_id,
                'filename': file.name,
                'insights': insights
            }
            response_serializer = AnalysisResultSerializer(response_data)
            return Response(response_serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class DocumentListView(generics.ListAPIView):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer

class DocumentDetailView(generics.RetrieveAPIView):
    queryset = Document.objects.all()
    serializer_class = DocumentDetailSerializer
    lookup_field = 'doc_id'

class ComplianceInsightListView(generics.ListAPIView):
    queryset = ComplianceInsight.objects.all()
    serializer_class = ComplianceInsightSerializer

    def get_queryset(self):
        queryset = super().get_queryset()
        doc_id = self.request.query_params.get('doc_id')
        framework = self.request.query_params.get('framework')
        if doc_id:
            queryset = queryset.filter(document__doc_id=doc_id)
        if framework:
            queryset = queryset.filter(framework__name=framework)
        return queryset

class RegulatoryFrameworkListView(generics.ListAPIView):
    queryset = RegulatoryFramework.objects.all()
    serializer_class = RegulatoryFrameworkSerializer

class RequirementListView(generics.ListAPIView):
    queryset = Requirement.objects.all()
    serializer_class = RequirementSerializer

    def get_queryset(self):
        queryset = super().get_queryset()
        framework = self.request.query_params.get('framework')
        if framework:
            queryset = queryset.filter(framework__name=framework)
        return queryset
    


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

class ProductInfoUploadView(APIView):
    def post(self, request):
        serializer = ProductInfoUploadSerializer(data=request.data)
        if serializer.is_valid():
            text = serializer.validated_data['product_info']
            frameworks = serializer.validated_data['frameworks']
            doc_id = str(uuid.uuid4())
            doc_title = f"Product_Info_{doc_id[:8]}"
            doc = Document.objects.create(doc_id=doc_id, title=doc_title)
            
            # Store text in MongoDB
            try:
                mongo_client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
                mongo_db = mongo_client['regulatory_mongo']
                mongo_db.documents.insert_one({'doc_id': doc_id, 'content': text})
            except Exception as e:
                return Response(
                    {'error': f"Failed to store document in MongoDB: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            finally:
                if 'mongo_client' in locals():
                    mongo_client.close()

            # Create FrameworkSelection entries
            for framework in frameworks:
                FrameworkSelection.objects.create(document=doc, framework=framework)

            # Analyze the text using the provided analyze_document function
            try:
                insights = analyze_document(text, frameworks, doc_id)
            except Exception as e:
                return Response(
                    {'error': f"Analysis failed: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # Store analysis results
            for insight in insights:
                framework_obj, _ = RegulatoryFramework.objects.get_or_create(name=insight['framework'])
                insight_obj = ComplianceInsight.objects.create(
                    document=doc,
                    framework=framework_obj,
                    insight=json.dumps(insight)
                )
                if insight.get('key_requirements'):
                    for req in insight['key_requirements']:
                        req_id = f"{framework_obj.name}_{req[:10].replace(' ', '_')}"
                        req_obj, _ = Requirement.objects.get_or_create(
                            framework=framework_obj,
                            requirement_id=req_id,
                            defaults={'description': req, 'category': 'General' }
                        )
                        insight_obj.related_requirements.add(req_obj)

            # Update FrameworkSelection status
            for framework in frameworks:
                FrameworkSelection.objects.filter(document=doc, framework=framework).update(analysis_status='COMPLETED')

            # Update risk classification for EU_AI_ACT
            for insight in insights:
                if insight['framework'] == 'EU_AI_ACT':
                    doc.risk_classification = insight.get('risk_classification', 'Not Classified')
                    doc.save()
                    break

            # Prepare response
            response_data = {
                'doc_id': doc_id,
                'filename': doc_title,
                'insights': insights
            }
            response_serializer = AnalysisResultSerializer(response_data)
            return Response(response_serializer.data, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)