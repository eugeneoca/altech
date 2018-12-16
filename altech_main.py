from altechlibrary import *
from altech_gui import *
import form_RegisterIdentity as frm_register
import form_TrainIdentity as frm_train

class Main(QtWidgets.QMainWindow, Ui_MainWindow):

	util = Utility()
	classifier = Classifier("altech_model")
	camera_array = []
	camWindow_array = []
	registration_process_flag = 1 # stop when 0
	training_process_flag = 0 # stop when 0
	# Initiate Classifier (Random Forest Classifier)
	classifier.load()
	authorized_upload = ""
	unauthorized_upload = ""
	
	def __init__(self):
		super(Main, self).__init__()
		self.setupUi(self)

		# Register Camera Windows
		self.camWindow_array.append(self.camWindow1)
		self.camWindow_array.append(self.camWindow2)
		self.camWindow_array.append(self.camWindow3)
		self.camWindow_array.append(self.camWindow4)
		self.camWindow_array.append(self.camWindow5)
		self.camWindow_array.append(self.camWindow6)
		self.camWindow_array.append(self.camWindow7)
		self.camWindow_array.append(self.camWindow8)

		# Initiate File Uploads
		self.authorized_upload = Internet("https://www.googleapis.com/auth/drive", "1cGCwCROlh9SPej5dD1vwMS5f3TtDFAlk", "token.json", "credentials.json")
		self.unauthorized_upload = Internet("https://www.googleapis.com/auth/drive", "1ddgNaBrlxGOgJynsNVgZ9Rjqk2Xo_fx4", "token.json", "credentials.json")

		self.authorized_upload.setDirectory("AUTHORIZED")
		self.unauthorized_upload.setDirectory("UNAUTHORIZED")

		# UPLOAD MONITORS
		self.util.runOnThread(name="NULL", method=self.monitorAuthUpload)
		self.util.runOnThread(name="NULL", method=self.monitorUnAuthUpload)

		# Events
		self.btn_RegisterIdentity.clicked.connect(self.show_RegistrationForm)
		self.btn_TrainIdentity.clicked.connect(self.show_TrainIdentityForm)

		# Run on thread
		self.util.runOnThread(name="Camera Update", method=self.updateCameraArray)
		self.util.runOnThread(name="Camera Feed Cycle", method=self.cameraFeedCycle)

	def show_RegistrationForm(self):
		self.window = QtWidgets.QMainWindow()
		self.form_registerProperties = frm_register.Ui_MainWindow()
		self.ui = self.form_registerProperties.setupUi(self.window)
		for i, cam in enumerate(self.camera_array):
			prop = self.util.getCamProperties(i, cam)
			self.form_registerProperties.cmb_CameraList.addItem(prop)
		self.form_registerProperties.txt_fname.textChanged.connect(self.updateDirectoryTextBox)
		self.form_registerProperties.txt_lname.textChanged.connect(self.updateDirectoryTextBox)
		self.form_registerProperties.btn_startRegistration.clicked.connect(self.startRegistration)
		self.form_registerProperties.btn_stopRegistration.clicked.connect(self.stopRegistration)
		self.window.show()

	def show_TrainIdentityForm(self):
		self.window = QtWidgets.QMainWindow()
		self.form_trainProperties = frm_train.Ui_MainWindow()
		self.ui = self.form_trainProperties.setupUi(self.window)
		identityCount = self.util.countContent("dataset")
		identity_folders = self.util.get_FileContents("dataset")
		total_samples = 0
		for folder in identity_folders:
			total_samples += self.util.countContent("dataset\\"+folder)

		self.form_trainProperties.txt_TotalIdentities.setText(str(identityCount) + " Total Identities | " + str(total_samples) + " Total samples")
		self.form_trainProperties.btn_startTraining.clicked.connect(self.startTraining)
		self.window.show()

	def startTraining(self):
		if self.training_process_flag==0:
			self.util.runOnThread(name="NULL", method=self.training_process)
			self.form_trainProperties.btn_startTraining.setText("Stop Training")
			self.training_process_flag=1
		else:
			self.form_trainProperties.btn_startTraining.setText("Start Training")
			self.training_process_flag=0

	def training_process(self):
		self.form_trainProperties.txt_TrainingProcess.setText("Creating Instance...")
		trainer = Trainer("altech_model")
		self.form_trainProperties.txt_TrainingProcess.setText("Initiating...")
		trainer.initiate()
		self.form_trainProperties.txt_TrainingProcess.setText("Importing datasets...")
		x, y = trainer.extract_fromDirectory('dataset')
		self.form_trainProperties.txt_TrainingProcess.setText("Training...")
		trainer.train(x, y)
		self.form_trainProperties.txt_TrainingProcess.setText("Saving...")
		trainer.save()
		self.form_trainProperties.txt_TrainingProcess.setText("Done...")
		self.form_trainProperties.btn_startTraining.click()


	def updateDirectoryTextBox(self):
		fname = self.form_registerProperties.txt_fname.text().lower()
		lname = self.form_registerProperties.txt_lname.text().lower()
		directory = "dataset\\"+fname+"-"+lname
		self.form_registerProperties.txt_directory.setText(directory)

	def stopRegistration(self):
		self.registration_process_flag=0

	def startRegistration(self):
		self.registration_process_flag=1
		fname = self.form_registerProperties.txt_fname.text()
		lname = self.form_registerProperties.txt_lname.text()
		if fname=="" or lname=="":
			self.form_registerProperties.txt_process.setText("Failed to start. Please click stop.")
			return
		dataset_name = fname.lower()+'-'+lname.lower()

		directory = self.util.setDirectory('dataset\\'+str(dataset_name))
		content_count = self.util.countContent(directory)
		self.util.runOnThread(name="NULL", method=self.registration_process, args=[directory, content_count])

	def registration_process(self, directory, initial_count):
		index = self.form_registerProperties.cmb_CameraList.currentIndex()
		id_proc = str(np.random.randint(10000)+1)
		i = initial_count
		while True:
			ret, frame = self.camera_array[index].readFrame()
			gray = self.util.toGray(frame)
			faces = self.util.faceDetect(gray)
			frame_withFaces = self.util.drawFaceRectangle(frame, faces)
			cv.imshow('Collecting Faces ('+ id_proc +')', frame_withFaces)

			# Ready for dataset creation
			for (x,y,w,h) in faces:
				roi_gray = gray[y:y+h, x:x+w]
				sample = cv.resize(roi_gray, (100,100))
				sample = cv.adaptiveThreshold(sample,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
				self.form_registerProperties.txt_process.setText(str(i)+" Gathered samples")
				cv.imwrite(directory+'\\sample_'+str(i)+".jpg", sample)
				i+=1

			if self.registration_process_flag==0:
				self.form_registerProperties.btn_stopRegistration.click()
				break

			if cv.waitKey(1) & 0xFF == ord('q'):
				self.form_registerProperties.btn_stopRegistration.click()
				break

	def updateCameraArray(self):
		while True:
			if self.classifier.ready==False:
				continue
			lock = threading.Lock()
			lock.acquire()
			cam_count = self.util.countCamera()
			#cam_count = 8
			if cam_count != len(self.camera_array):
				for i,cam in enumerate(self.camera_array):
					self.camWindow_array[i].clear()
					cam.release()
					self.camera_array = []
				for cam_id in range(cam_count):
					cam = Camera('CAM '+str(cam_id), cam_id)
					#cam = Camera('CAM '+str(cam_id), "rtsp://admin:1234abcd@192.168.1.64:554/PSIA/streaming/channels/"+str(cam_id+1)+"02?network-caching=0")
					self.camera_array.append(cam)

				self.util.log("[MAIN]", str(self.camera_array))

			lock.release()

	def cameraFeedCycle(self):
		while True:
			if self.classifier.ready==False:
				continue
			lock = threading.Lock()
			lock.acquire()

			for i,cam in enumerate(self.camera_array):
				try:
					# Error handling
					ret, frame = cam.readFrame()
					
					faces = self.util.faceDetect(self.util.toGray(frame))
					frame_rect = self.util.drawFaceRectangle(frame, faces)
					self.util.setPictureImage(self.camWindow_array[i], frame)
					for face in faces:
						roi = self.util.getFaceFrame(frame, face)
						filtered = self.util.filterAdaptive(roi)
						resized = self.util.resize(filtered, (100, 100))
						flat = self.util.flat(resized)
						self.util.runOnThread(name="NULL", method=self.classifier.predict, args=[flat, frame])
						#self.util.runOnThread(name="NULL", method=self.monitorResult)

				except Exception as err:
					print(cam.name + ": " + str(err))

			if cv.waitKey(1) & 0xFF == ord('q'):
				break

			lock.release()

	def monitorResult(self):
		if self.classifier.ready==False:
			return
		lock = threading.Lock()
		lock.acquire()
		result = self.classifier.result
		if result == "Unknown":
			#print(dir(self.txt_log))
			self.txt_log.append(result + " identity has been detected.")
		else:
			#print(dir(self.txt_log))
			self.txt_log.append(result + " identity has been detected.")
		lock.release()

	def monitorAuthUpload(self):
		while True:
			if self.classifier.ready==False:
				continue
			lock = threading.Lock()
			lock.acquire()
			try:
				for image in self.classifier.auth_upload:
					self.authorized_upload.upload(image)
					self.classifier.auth_upload.remove(image)
			except Exception as error:
				print("")
			lock.release()

	def monitorUnAuthUpload(self):
		while True:
			if self.classifier.ready==False:
				continue
			lock = threading.Lock()
			lock.acquire()
			for image in self.classifier.unauth_upload:
				self.unauthorized_upload.upload(image)
				self.classifier.unauth_upload.remove(image)
			lock.release()

app = QtWidgets.QApplication([])
application = Main()
application.show()
sys.exit(app.exec())