# Generated by Django 5.0.3 on 2024-04-19 16:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mainapp', '0020_cluster_records_inertia'),
    ]

    operations = [
        migrations.AlterField(
            model_name='globals',
            name='training_thread_running',
            field=models.BooleanField(default=False),
        ),
    ]
