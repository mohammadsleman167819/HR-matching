# Generated by Django 5.0.3 on 2024-04-22 05:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mainapp', '0023_alter_cluster_records_options_alter_course_options_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='job_post',
            name='jobDescription',
            field=models.TextField(max_length=5000, verbose_name='Job Description'),
        ),
    ]
