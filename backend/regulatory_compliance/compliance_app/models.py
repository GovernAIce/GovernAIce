from django.db import models

class Document(models.Model):
    doc_id = models.CharField(max_length=100, unique=True)
    title = models.CharField(max_length=255)
    upload_date = models.DateTimeField(auto_now_add=True)
    risk_classification = models.CharField(max_length=50, blank=True)

    def __str__(self):
        return self.title

class RegulatoryFramework(models.Model):
    name = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True)
    version = models.CharField(max_length=20, blank=True)

    def __str__(self):
        return self.name

class Requirement(models.Model):
    framework = models.ForeignKey(RegulatoryFramework, on_delete=models.CASCADE, related_name='requirements')
    requirement_id = models.CharField(max_length=50, unique=True)
    description = models.TextField()
    category = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return f"{self.requirement_id} - {self.framework.name}"

class FrameworkSelection(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    framework = models.CharField(max_length=50, choices=[
        ('EU_AI_ACT', 'EU AI Act'),
        ('NIST_RMF', 'NIST AI RMF'),
        ('CPRA', 'CPRA'),
    ])
    analysis_status = models.CharField(max_length=50, default='PENDING')

class ComplianceInsight(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    framework = models.ForeignKey(RegulatoryFramework, on_delete=models.CASCADE)
    insight = models.TextField()  # JSON string
    created_at = models.DateTimeField(auto_now_add=True)
    related_requirements = models.ManyToManyField(Requirement, blank=True)

    def __str__(self):
        return f"Insight for {self.document} - {self.framework}"