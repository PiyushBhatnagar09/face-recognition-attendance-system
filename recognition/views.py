from django.shortcuts import render,redirect

#all forms
from .forms import usernameForm,DateForm,UsernameAndDateForm, DateForm_2
from django.contrib import messages
from django.contrib.auth.models import User
import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner
import time
from attendance_system_facial_recognition.settings import BASE_DIR
import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from django.contrib.auth.decorators import login_required
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime
from django_pandas.io import read_frame
from users.models import Present, Time
import seaborn as sns
import pandas as pd
from django.db.models import Count
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams
import math

mpl.use('Agg')

#checks username entered by user is correct or not
def username_present(username):
	if User.objects.filter(username=username).exists():
		return True
	return False

#create dataset of a particular user: We are only converting the image to grayscale because it is then easy to detect face in the image and then we crop the original image such that only face is visible.
def create_dataset(username):
	id = username

	#path for storing the dataset(if path not present then create one)
	if(os.path.exists('face_recognition_data/training_dataset/{}/'.format(id))==False):
		os.makedirs('face_recognition_data/training_dataset/{}/'.format(id))

	directory='face_recognition_data/training_dataset/{}/'.format(id)

	#'detector' will detect the faces in the given grayscale image and will give us the coordinate where the faces are located
	detector = dlib.get_frontal_face_detector()
	#'predictor' will give the coordinates of all the 68 face landmarks in the face in the image
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
	#face aligner is used to align or normalize face into a standardized position or orientation
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	#accessing pc's camera
	vs = VideoStream(src=0).start()

	sampleNum = 0
	# Capturing the faces one by one
	while(True):
		#capturing image from our camera
		frame = vs.read()
		
		#resize and convert the image to grayscale because all ML models works on grayscale images because it is easy to work on them
		frame = imutils.resize(frame ,width = 800)
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		#This will detect all the faces in the current frame, and it will return the coordinates of the faces
		faces = detector(gray_frame,0)
		#In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.
		
		for face in faces:
			#rectangle to bounding box
			"""
			we normally think of a bounding box in terms of “(x, y, width, height)” so as a matter of convenience, 
			the rect_to_bb function takes this rect object and transforms it into a 4-tuple of coordinates.
			"""
			(x,y,w,h) = face_utils.rect_to_bb(face)
			
			#aligning or normalizing the face into standardized position or orientation ('fa' is the instance of FaceAligner we made)
			face_aligned = fa.align(frame,gray_frame,face)

			sampleNum = sampleNum+1
			
			if face is None:
				continue

			cv2.imwrite(directory+'/'+str(sampleNum)+'.jpg'	, face_aligned)
			face_aligned = imutils.resize(face_aligned ,width = 400)

			#drawing a rectangle around the face in the colored image
			cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 1)
			
			#for a little pause
			cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow("Add Images",frame)

		#Before closing it we need to give a wait command, otherwise the open cv wont work
		cv2.waitKey(1)

		#To get out of the loop
		if(sampleNum>100):
			break
	
	#Stoping the videostream
	vs.stop()
	# destroying all the windows
	cv2.destroyAllWindows()


def predict(face_aligned, svc, threshold=0.7):
	face_encodings = np.zeros((1,128))

	"""
	Face encodings are numerical representations of facial features that can be used for face recognition tasks. The face_recognition.face_encodings function is used to extract these encodings from the aligned face image.
	"""
	try:
		x_face_locations=face_recognition.face_locations(face_aligned)
		faces_encodings=face_recognition.face_encodings(face_aligned,known_face_locations=x_face_locations)

		#if no face is detected
		if(len(faces_encodings)==0):
			return ([-1],[0])

	except:
		return ([-1],[0])

	prob=svc.predict_proba(faces_encodings)

	#finding the index of the class with highest probability
	result=np.where(prob[0]==np.amax(prob[0]))

	#if highest prob. is less or equal to threshold then no valid prediction is made
	if(prob[0][result[0]]<=threshold):
		return ([-1],prob[0][result[0]])

	#returning predicted class index and it's probability
	return (result[0],prob[0][result[0]])


def vizualize_Data(embedded, targets,): #Draw scatter plot
	# print("Embedded: ", embedded)
	# print("Targets: ", targets)
	"""
	The embedded data points are transformed into a two-dimensional space using t-Distributed Stochastic Neighbor Embedding (t-SNE) with two components. 
	"""
	X_embedded = TSNE(n_components=2).fit_transform(embedded)
	# print("X_embedded: ", X_embedded)
 
	for i, t in enumerate(set(targets)): #t's value is [piyush_0902, abhinav_07, vinita_20] one by one because we are using 'set' (unique values)
		idx = targets == t #idx is an array of (true, false) denoting which entry in target array matches with 't'
		# print("idx, i : ", idx, i)
		plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

	plt.legend(bbox_to_anchor=(1, 1))
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()	
	plt.savefig('./recognition/static/recognition/img/training_visualisation.png')
	plt.close()


def update_attendance_in_db_in(present):
    today = datetime.date.today()
    time = datetime.datetime.now()
    # print('CHECK: ', present)
    for person in present:
        user = User.objects.get(username=person)
        try:
            qs = Present.objects.get(user=user, date=today)
        except Present.DoesNotExist:
            qs = None
        
        if qs is None:
            if present[person] == True:
                a = Present(user=user, date=today, present=True)
                a.save()
            else:
                a = Present(user=user, date=today, present=False)
                a.save()
        else:
            if present[person] == True:
                qs.present = True
                qs.save(update_fields=['present'])
        
        if present[person] == True:
            a = Time(user=user, date=today, time=time, out=False)
            a.save()

def update_attendance_in_db_out(present):
	today=datetime.date.today()
	time=datetime.datetime.now()
	for person in present:
		user=User.objects.get(username=person)
		if present[person]==True:
			a=Time(user=user,date=today,time=time, out=True)
			a.save()

def convert_hours_to_hours_mins(hours):
	h=int(hours)
	hours-=h
	m=hours*60
	m=math.ceil(m)
	return str(str(h)+ " hrs " + str(m) + "  mins")

#-----------------------------------------------------------------Display Graphs---------------------------------------------------------
def hours_vs_date_given_employee(present_qs,time_qs,admin=True):
	register_matplotlib_converters()
	df_hours=[]
	qs=present_qs

	for obj in qs:
		date=obj.date
		times_in=time_qs.filter(date=date).filter(out=False).order_by('time')
		times_out=time_qs.filter(date=date).filter(out=True).order_by('time')

		obj.time_in=None
		obj.time_out=None
		obj.hours=0
		if (len(times_in)>0):			
			obj.time_in=times_in.first().time
			
		if (len(times_out)>0):
			obj.time_out=times_out.last().time

		if(obj.time_in is not None and obj.time_out is not None):
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
			obj.hours=hours
		else:
			obj.hours=0
		
		df_hours.append(obj.hours)
		obj.hours=convert_hours_to_hours_mins(obj.hours)
			
	df = read_frame(qs)		
	df["hours"]=df_hours

	plt.figure(figsize=(12, 6))
	sns.barplot(data=df,x='date',y='hours')
	plt.xticks(rotation='horizontal')
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	
	if(admin):
		plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_date/1.png')
		plt.close()
	else:
		plt.savefig('./recognition/static/recognition/img/attendance_graphs/employee_login/1.png')
		plt.close()
	return qs
	
def hours_vs_employee_given_date(present_qs,time_qs):
	#present_qs, time_qs are objects for a particular date
	register_matplotlib_converters()
	df_hours=[]
	df_username=[]
	qs=present_qs

	for obj in qs:
		user=obj.user
		times_in=time_qs.filter(user=user).filter(out=False)
		times_out=time_qs.filter(user=user).filter(out=True)

		obj.time_in=None
		obj.time_out=None
		obj.hours=0
		obj.hours=0

		if (len(times_in)>0):			
			obj.time_in=times_in.first().time
		if (len(times_out)>0):
			obj.time_out=times_out.last().time
		
		#calculating the total no. of hours user worked
		if(obj.time_in is not None and obj.time_out is not None):
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
			obj.hours=hours
		else:
			obj.hours=0
		
		df_hours.append(obj.hours)
		df_username.append(user.username)
		obj.hours=convert_hours_to_hours_mins(obj.hours)

	df = read_frame(qs)	
	df["hours"]=df_hours
	df["username"]=df_username

	plt.figure(figsize=(12, 6))
	sns.barplot(data=df,x='username',y='hours')
	plt.xticks(rotation='horizontal')
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_employee/1.png')
	plt.close()
	return qs


def total_number_employees():
	qs=User.objects.all()
	return (len(qs)-1) #we subtract 1 because we don't count admin as employee

def employees_present_today():
	today=datetime.date.today()
	qs=Present.objects.filter(date=today).filter(present=True)
	return len(qs)


def this_week_emp_count_vs_date():
	today=datetime.date.today()
	some_day_last_week= today - datetime.timedelta(days=7)
	monday_of_last_week=some_day_last_week - datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
	monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
	
	#qs is queryset
	qs=Present.objects.filter(date__gte=monday_of_this_week).filter(date__lte=today) #date greater than equal to, less than equal to
	str_dates=[]
	emp_count=[]
	str_dates_all=[]
	emp_cnt_all=[]
	cnt=0

	for obj in qs:
		date=obj.date
		str_dates.append(str(date))
		qs=Present.objects.filter(date=date).filter(present=True)
		emp_count.append(len(qs))

	#finding the no. of employees present on each date starting from 'monday of this week'
	while(cnt<7):
		date=str(monday_of_this_week+datetime.timedelta(days=cnt))
		cnt+=1
		str_dates_all.append(date)
		if(str_dates.count(date))>0:
			idx=str_dates.index(date)
			emp_cnt_all.append(emp_count[idx])
		else:
			emp_cnt_all.append(0)

	df=pd.DataFrame()
	df["Date"]=str_dates_all
	df["Number of employees"]=emp_cnt_all
	
	plt.figure(figsize=(12,6))
	sns.lineplot(data=df,x='Date',y='Number of employees')
	plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
	plt.tight_layout()  # Adjust layout to prevent overlapping elements
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/this_week/1.png')
	plt.close()

def last_week_emp_count_vs_date():
	today=datetime.date.today()
	some_day_last_week=today-datetime.timedelta(days=7)
	monday_of_last_week=some_day_last_week-  datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
	monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)

	qs=Present.objects.filter(date__gte=monday_of_last_week).filter(date__lt=monday_of_this_week)
	str_dates=[]
	emp_count=[]
	str_dates_all=[]
	emp_cnt_all=[]
	cnt=0
	
	for obj in qs:
		date=obj.date
		str_dates.append(str(date))
		qs=Present.objects.filter(date=date).filter(present=True)
		emp_count.append(len(qs))

	while(cnt<7):
		date=str(monday_of_last_week+datetime.timedelta(days=cnt))
		cnt+=1
		str_dates_all.append(date)
		if(str_dates.count(date))>0:
			idx=str_dates.index(date)
			emp_cnt_all.append(emp_count[idx])
		else:
			emp_cnt_all.append(0)

	df=pd.DataFrame()
	df["Date"]=str_dates_all
	df["Number of employees"]=emp_cnt_all
	plt.figure(figsize=(12,6))
	sns.lineplot(data=df,x='Date',y='Number of employees')
	plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
	plt.tight_layout()  # Adjust layout to prevent overlapping elements
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/last_week/1.png')
	plt.close()

#-------------------------------------------------------------------------------------------------------------------------------------

def home(request):
	return render(request, 'recognition/home.html')

@login_required
def dashboard(request):
	if(request.user.username=='admin'):
		return render(request, 'recognition/admin_dashboard.html')
	else:
		return render(request,'recognition/employee_dashboard.html')

@login_required
def add_photos(request):
	if request.user.username!='admin':
		return redirect('not-authorised')

	if request.method=='POST':
		form=usernameForm(request.POST)
		data = request.POST.copy()
		username=data.get('username')
		if username_present(username):
			#take images of user and create dataset by marking all the 68 facial landmarks on face
			# print('Username: ', username)
			create_dataset(username)
			messages.success(request, f'Dataset Created')
			return redirect('add-photos')
		else:
			messages.warning(request, f'No such username found. Please register employee first.')
			return redirect('dashboard')
	else:
			form=usernameForm()
			return render(request,'recognition/add_photos.html', {'form' : form})

def mark_your_attendance(request):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
	
	#using our trained svc model for classification
	svc_save_path="face_recognition_data/svc.sav"	

	with open(svc_save_path, 'rb') as f:
		svc = pickle.load(f)
	
	#face aligner is used to standardize the face i.e. it's width, etc. So that when we get a new image to classify, we will standardize it also and then predict
	fa = FaceAligner(predictor , desiredFaceWidth = 96)

	#during training we encoded the y labels into numerical data, now we will inverse_tranform it to get our labels back
	encoder=LabelEncoder()
	#classes.npy stores the y labels i.e. classes. Ex: [piyush_0902]
	encoder.classes_ = np.load('face_recognition_data/classes.npy')

	#we take out all the faces that we can predict from our svc model
	faces_encodings = np.zeros((1,128))
	no_of_faces = len(svc.predict_proba(faces_encodings)[0])

	count = dict()
	present = dict()
	log_time = dict()
	start = dict()
	for i in range(no_of_faces):
		#count['piyush_0902']=0
		count[encoder.inverse_transform([i])[0]] = 0
		#present['piyush_0902']=0
		present[encoder.inverse_transform([i])[0]] = False

	vs = VideoStream(src=0).start()
	sampleNum = 0
	while(True):
		#detect faces from camera and convert it to grayscale image
		frame = vs.read()
		frame = imutils.resize(frame, width = 800)
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = detector(gray_frame,0)
		
		for face in faces:
			(x,y,w,h) = face_utils.rect_to_bb(face)

			face_aligned = fa.align(frame,gray_frame,face)

			#we made a green color rectangle on your face with thickness 1
			cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 1)
					
			#calculate the prediction and probability of face identity using the svc model
			#pred is the class in which the face is classified but pred is in numerical format, which we got using encoder, so we have to use inverse_transform to get actual class name
			(pred, prob)=predict(face_aligned, svc)
			
			if(pred!=[-1]): #if valid prediction then
				person_name=encoder.inverse_transform(np.ravel([pred]))[0]
				pred=person_name
				if count[pred] == 0:
					start[pred] = time.time()
					count[pred] = count.get(pred,0) + 1

				if count[pred] == 4 and (time.time() - start[pred]) > 1.2:
					count[pred] = 0
				else:
					present[pred] = True
					log_time[pred] = datetime.datetime.now()
					count[pred] = count.get(pred,0) + 1
					# print(pred, present[pred], count[pred])

				#we put person name + probability in the camera window below the green rectangle
				cv2.putText(frame, str(person_name)+ str(prob), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			else:
				person_name="unknown"
				cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

		#Showing the image in another window
		cv2.imshow("Mark Attendance - In - Press q to exit",frame)

		key=cv2.waitKey(50) & 0xFF
		if(key==ord("q")):
			break
	
	#Stoping the videostream
	vs.stop()

	# destroying all the windows
	cv2.destroyAllWindows()
	# print('PRESENT: ', present)
	update_attendance_in_db_in(present)
	return redirect('home')

def mark_your_attendance_out(request):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
	svc_save_path="face_recognition_data/svc.sav"	

	with open(svc_save_path, 'rb') as f:
			svc = pickle.load(f)

	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	encoder=LabelEncoder()
	encoder.classes_ = np.load('face_recognition_data/classes.npy') #stores the y labels i.e. person names in numerical format

	faces_encodings = np.zeros((1,128)) #empty 2D array with 1 row and 128 columns
	no_of_faces = len(svc.predict_proba(faces_encodings)[0]) #total number of faces that our model can predict
	count = dict()
	present = dict()
	log_time = dict()
	start = dict()

	""" TO UNDERSTAND HOW inverse_transform works :-
	>>> le.transform([1, 1, 2, 6])
	array([0, 0, 1, 2]...)
	>>> le.inverse_transform([0, 0, 1, 2])
	array([1, 1, 2, 6])
	"""
	for i in range(no_of_faces): #i is index
		#count of each face is initially set to 0
		count[encoder.inverse_transform([i])[0]] = 0
		#each face is initially marked as absent i.e. false
		present[encoder.inverse_transform([i])[0]] = False

	vs = VideoStream(src=0).start()
	
	while(True):
		frame = vs.read()
		frame = imutils.resize(frame ,width = 800)
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		#take faces from the camera
		faces = detector(gray_frame,0)
	
		for face in faces:
			(x,y,w,h) = face_utils.rect_to_bb(face)
			face_aligned = fa.align(frame,gray_frame,face) #standardizing the face image
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1) #green rectangle
					
			(pred,prob)=predict(face_aligned, svc) #predicted person name(numerical format) and probability
			
			if(pred!=[-1]):
				person_name=encoder.inverse_transform(np.ravel([pred]))[0]
				pred=person_name

				if count[pred] == 0:
					start[pred] = time.time()
					count[pred] = count.get(pred,0) + 1

				if count[pred] == 4 and (time.time()-start[pred]) > 1.5:
					count[pred] = 0
				else:
					present[pred] = True
					log_time[pred] = datetime.datetime.now()
					count[pred] = count.get(pred,0) + 1
					# print(pred, present[pred], count[pred])

				cv2.putText(frame, str(person_name)+ str(prob), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			else:
				person_name="unknown"
				cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

		#Showing the image in another window
		cv2.imshow("Mark Attendance- Out - Press q to exit",frame)
		
		key=cv2.waitKey(50) & 0xFF
		if(key==ord("q")):
			break
	
	#Stoping the videostream
	vs.stop()

	# destroying all the windows
	cv2.destroyAllWindows()
	update_attendance_in_db_out(present)
	return redirect('home')


#---------------------------------------------------------------------ADMIN-------------------------------------------------------------------------
@login_required
def train(request):
	#only admin can train the model
	if request.user.username!='admin':
		return redirect('not-authorised')

	training_dir='face_recognition_data/training_dataset'
	
	X=[]
	y=[]
	i=0

	for person_name in os.listdir(training_dir):
		# print(str(person_name))
		curr_directory=os.path.join(training_dir, person_name)
		if not os.path.isdir(curr_directory):
			continue

		for imagefile in image_files_in_folder(curr_directory):
			# print(str(imagefile))
			image=cv2.imread(imagefile)
			try:
				"""
				Facial encoding of an image refers to the process of representing a face as a numerical vector or array. 
				It involves extracting distinctive features from the facial image that can be used to uniquely identify or 
				characterize that particular face. These features typically capture details such as the position of key facial 
				landmarks, variations in skin tone, texture, and other facial attributes.
				"""
				#extracts facial encoding from the image and convert it to list
				X.append((face_recognition.face_encodings(image)[0]).tolist())
				y.append(person_name)
				i+=1
			except:
				# print("Removed Image")
				os.remove(imagefile)

	# print('X: ', X)
	# print('y: ', y)
	targets=np.array(y)
	# print(targets)

	"""
	The LabelEncoder is a utility class provided by the scikit-learn library in Python. 
	Its primary purpose is to encode categorical labels (textual or string-based labels) into numerical representations. 
	This transformation is essential in machine learning tasks because many algorithms require numerical input.
	
	Fit: You then use the fit() method of the LabelEncoder instance to fit the encoder to your list of categorical labels. For example, if you have a list of names like ["Alice", "Bob", "Charlie"], you would call encoder.fit(["Alice", "Bob", "Charlie"]).

	Transform: After fitting the encoder, you can use the transform() method to transform your original labels into numerical representations. For instance, if you call encoder.transform(["Alice", "Bob", "Charlie"]), you might get an output like [0, 1, 2], where each name has been assigned a unique numerical value.

	Inverse Transform: If needed, you can also use the inverse_transform() method to convert the numerical representations back into their original categorical labels. For example, calling encoder.inverse_transform([0, 1, 2]) might return ["Alice", "Bob", "Charlie"]
	"""
	encoder = LabelEncoder()
	encoder.fit(y)
	y=encoder.transform(y)
	X1=np.array(X)
	# print(X1)
	# print("shape: "+ str(X1.shape))
	# print('Encoder classes: ', encoder.classes_)
 
	#encoder.classes_ stores information related to which label is represented as which numerical representation, so that we can inverse transform it later on
	np.save('face_recognition_data/classes.npy', encoder.classes_)

	# N O T E: svc model requires atleast two classes to make the hyperplane to classify input into classes.
	"""
	Here's what a linear kernel does:

	Linear Separation: A linear kernel essentially performs a linear transformation on the input data. It calculates the dot product between pairs of samples in the input space.

	Hyperplane: In the transformed space, the linear kernel aims to find a hyperplane that separates the data points of different classes with the maximum margin. 
	This hyperplane is a decision boundary that separates the classes as cleanly as possible.
	"""
	#see the link: https://www.geeksforgeeks.org/creating-linear-kernel-svm-in-python/ to understand linear kernel(basically to separate the classes using linear lines)
	svc = SVC(kernel='linear', probability=True)
	svc.fit(X1,y) #fitting our svc model with the dataset
	svc_save_path="face_recognition_data/svc.sav"
	with open(svc_save_path, 'wb') as f:
		pickle.dump(svc,f)
	
	vizualize_Data(X1,targets)
	
	messages.success(request, f'Training Complete.')
	return render(request,"recognition/train.html")

@login_required
def not_authorised(request):
	return render(request,'recognition/not_authorised.html')

@login_required
def view_attendance_home(request):
	total_num_of_emp=total_number_employees()
	emp_present_today=employees_present_today()
	this_week_emp_count_vs_date()
	last_week_emp_count_vs_date()
	return render(request,"recognition/view_attendance_home.html", {'total_num_of_emp' : total_num_of_emp, 'emp_present_today': emp_present_today})

@login_required
def view_attendance_date(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	
	qs=None
	time_qs=None
	present_qs=None

	if request.method=='POST':
		form=DateForm(request.POST)
		if form.is_valid():
			date=form.cleaned_data.get('date')
			time_qs=Time.objects.filter(date=date)
			present_qs=Present.objects.filter(date=date)

			if(len(time_qs)>0 or len(present_qs)>0):
				qs=hours_vs_employee_given_date(present_qs,time_qs)
				return render(request,'recognition/view_attendance_date.html', {'form' : form,'qs' : qs })
			else:
				messages.warning(request, f'No records for selected date.')
				return redirect('view-attendance-date')
	else:
			form=DateForm()
			return render(request,'recognition/view_attendance_date.html', {'form' : form, 'qs' : qs})

@login_required
def view_attendance_employee(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	
	time_qs=None
	present_qs=None
	qs=None

	if request.method=='POST':
		form=UsernameAndDateForm(request.POST)
		if form.is_valid():
			username=form.cleaned_data.get('username')

			if username_present(username):
				u=User.objects.get(username=username)
				time_qs=Time.objects.filter(user=u)
				present_qs=Present.objects.filter(user=u)

				date_from=form.cleaned_data.get('date_from')
				date_to=form.cleaned_data.get('date_to')
				
				if date_to < date_from:
					messages.warning(request, f'Invalid date selection.')
					return redirect('view-attendance-employee')
				else:
					time_qs=time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					present_qs=present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					
					if (len(time_qs)>0 or len(present_qs)>0):
						qs=hours_vs_date_given_employee(present_qs,time_qs,admin=True)
						return render(request,'recognition/view_attendance_employee.html', {'form' : form, 'qs' :qs})
					else:
						messages.warning(request, f'No records for selected duration.')
						return redirect('view-attendance-employee')
			else:
				messages.warning(request, f'No such username found.')
				return redirect('view-attendance-employee')
	else:
		form=UsernameAndDateForm()
		return render(request,'recognition/view_attendance_employee.html', {'form' : form, 'qs' :qs})


@login_required
def view_my_attendance_employee_login(request):
	if request.user.username=='admin':
		return redirect('not-authorised')
	
	qs=None
	time_qs=None
	present_qs=None

	if request.method=='POST':
		form=DateForm_2(request.POST)
		if form.is_valid():
			u=request.user
			time_qs=Time.objects.filter(user=u)
			present_qs=Present.objects.filter(user=u)

			date_from=form.cleaned_data.get('date_from')
			date_to=form.cleaned_data.get('date_to')

			if date_to < date_from:
					messages.warning(request, f'Invalid date selection.')
					return redirect('view-my-attendance-employee-login')
			else:
					time_qs=time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					present_qs=present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
				
					if (len(time_qs)>0 or len(present_qs)>0):
						qs=hours_vs_date_given_employee(present_qs,time_qs,admin=False)
						return render(request,'recognition/view_my_attendance_employee_login.html', {'form' : form, 'qs' :qs})
					else:
						
						messages.warning(request, f'No records for selected duration.')
						return redirect('view-my-attendance-employee-login')
	else:
			form=DateForm_2()
			return render(request,'recognition/view_my_attendance_employee_login.html', {'form' : form, 'qs' :qs})