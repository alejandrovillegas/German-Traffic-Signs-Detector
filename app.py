import urllib.request as req
import os
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score


@click.group()
def main():
    pass

@main.command("download")
def download():
    localFilePath = './images/dataset.zip'
    url = 'http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip'
    r = requests.get(url, allow_redirects=True)
    unzip(r)
    

def unzip(zfile, md=False):
	basedir = ''
	if md:
		basedir = prepareBaseDir(zfile)
	
	zfile = zipfile.ZipFile(zfile, 'r')
	for name in zfile.namelist():
		uname = name.decode('gbk')
		if uname.endswith('.DS_Store'):
			continue
		
		#prepare directory
		dirs = os.path.dirname(uname)
		if basedir:
			dirs = os.path.join(basedir, dirs)
		print 'extracting: ' + uname
		if dirs and not os.path.exists(dirs):
			print 'prepare directories: ', dirs
			os.makedirs(dirs)
		
		#ready to unzip file
		data = zfile.read(name)
		if basedir:
			uname = os.path.join(basedir, uname)
		if not os.path.exists(uname):
			fo = open(uname, 'w')
			fo.write(data)
			fo.close()
	zfile.close()

@main.command("scikitlearn")
def download():
    INIT_MODEL = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10)

class LogitSk():
  SAVE_PATH = os.path.abspath('models/model1/saved/logitsk.pkl')

  def __init__(self, model=INIT_MODEL):
    self.log_reg = model
  
  def train(self, x, y):
    print('Training Logistic Regression (sklearn) model...')
    self.log_reg.fit(x, y)
    self._save()
    self.calc_accuracy(x, y)
  
  def predict(self, data):
    return self.log_reg.predict(data)

  def _save(self):
    joblib.dump(self.log_reg, self.SAVE_PATH)
    print("Model persisted at", self.SAVE_PATH)
  
  def calc_accuracy(self, x, y):
    y_pred = self.log_reg.predict(x)
    accuracy = accuracy_score(y, y_pred) * 100
    print('Model accuracy {}%'.format(round(accuracy, 2)))
  
  @staticmethod
  def load_model():
    restored_model = joblib.load(LogitSk.SAVE_PATH)
    return LogitSk(model=restored_model)
  
