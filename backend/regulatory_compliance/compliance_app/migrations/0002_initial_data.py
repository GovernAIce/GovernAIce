from django.db import migrations

def load_initial_data(apps, schema_editor):
    RegulatoryFramework = apps.get_model('compliance_app', 'RegulatoryFramework')
    Requirement = apps.get_model('compliance_app', 'Requirement')

    frameworks = [
        {'name': 'EU_AI_ACT', 'description': 'EU AI Act 2024', 'version': '2024'},
        {'name': 'NIST_RMF', 'description': 'NIST AI Risk Management Framework', 'version': '1.0'},
        {'name': 'CPRA', 'description': 'California Privacy Rights Act', 'version': '2020'},
    ]
    for fw in frameworks:
        RegulatoryFramework.objects.create(**fw)

    requirements = [
        {'framework': 'EU_AI_ACT', 'requirement_id': 'EU_AI_ACT_HUMAN_OVERSIGHT', 'description': 'Human oversight for high-risk systems', 'category': 'Oversight'},
        {'framework': 'NIST_RMF', 'requirement_id': 'NIST_RMF_GOVERN', 'description': 'Establish governance policies', 'category': 'Governance'},
        {'framework': 'CPRA', 'requirement_id': 'CPRA_DATA_RIGHTS', 'description': 'Support data subject rights', 'category': 'Privacy'},
    ]
    for req in requirements:
        framework = RegulatoryFramework.objects.get(name=req['framework'])
        Requirement.objects.create(
            framework=framework,
            requirement_id=req['requirement_id'],
            description=req['description'],
            category=req['category']
        )

class Migration(migrations.Migration):
    dependencies = [
        ('compliance_app', '0001_initial'),
    ]
    operations = [
        migrations.RunPython(load_initial_data),
    ]