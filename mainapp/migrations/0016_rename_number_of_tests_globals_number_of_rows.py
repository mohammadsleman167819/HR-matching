# Generated by Django 5.0.3 on 2024-04-15 15:33

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mainapp', '0015_rename_test_thread_running_globals_testing_thread_running'),
    ]

    operations = [
        migrations.RenameField(
            model_name='globals',
            old_name='number_of_tests',
            new_name='number_of_rows',
        ),
    ]
