from django.db import models
from django.contrib.auth.models import User

import datetime
# Create your models here.
	
class Present(models.Model):
	user=models.ForeignKey(User,on_delete=models.CASCADE)
	date = models.DateField(default=datetime.date.today)
	present=models.BooleanField(default=False)
	
class Time(models.Model):
	user=models.ForeignKey(User,on_delete=models.CASCADE)
	date = models.DateField(default=datetime.date.today)
	time=models.DateTimeField(null=True,blank=True)
	out=models.BooleanField(default=False)

"""
user field is a foreign key to the User model, establishing a relationship between the Present model and the User model. 
The on_delete=models.CASCADE argument specifies that if a related user is deleted, all associated presence records should also be deleted.

date field represents the date of the presence record. It's a DateField initialized with the default value of today's date obtained from datetime.date.today().

present field is a boolean field indicating whether the user is present (True) or not (False) on the specified date. It defaults to False.
"""

"""
time field represents the actual time of entry or exit. It's a DateTimeField allowing both date and time values. 
It's nullable (null=True) and can be left blank (blank=True).

out field is a boolean field indicating whether the user has exited (True) or not (False). It defaults to False.
"""