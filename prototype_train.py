import numpy as np
import scipy.optimize as opt
import sys, os, random, gzip
import tensorflow as tf
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

# global flags & parameters
# remark: increase the sample size for full training, by default 2/3 (1/3) of the sample going to training (testing)
SRC_SIG = {'filename':'fatjet_w_match_vz_to_ww.txt.gz', 'skip':0, 'n':6000}
SRC_BKG = {'filename':'fatjet_q_match_vz_to_qq.txt.gz', 'skip':0, 'n':6000}
WEIGHTS_2PCNN = 'wgts_prototype.h5'
BATCH_SIZE = 80
EPOCHS = 20
ENABLE_EARLY_STOPPING = True
MAX_PATIENCE = 10

# constituents format
_index, _type, _pid, _charge, _pt, _eta, _phi, _vx, _vy, _vz = range(10)

def deltaPhi(phi1,phi2):
    x = phi1-phi2
    while x>= np.pi: x -= np.pi*2.
    while x< -np.pi: x += np.pi*2.
    return x

def deltaR(eta1,phi1,eta2,phi2):
    return (deltaPhi(phi1,phi2)**2+(eta1-eta2)**2)**0.5

def find_main_axis(clist):
    def fcn(p):
        dir_x,dir_y = np.cos(p[0]),np.sin(p[0])
        v = clist[(clist[:,_eta]**2+clist[:,_phi]**2)**0.5>1E-5]
        cosang = np.abs(dir_x*v[:,_eta]+dir_y*v[:,_phi])/(v[:,_eta]**2+v[:,_phi]**2)**0.5
        proj = (v[:,_pt]*cosang).sum()
        return -proj
    r = opt.minimize(fcn,[0.])
    if r.success: return r.x[0]
    else: return None

# reading jet data from the gzipped text stream
def parse_jet_data(fin):
    
    if '<jet_data>' not in fin.readline().decode():
        print(">>> ERROR: invalid input", flush=True)
        sys.exit(0)
    data = {}
    buf = fin.readline().decode().split() # jet kinematics
    data['index'] = int(buf[0])
    data['pt'] = float(buf[1])
    data['eta'] = float(buf[2])
    data['phi'] = float(buf[3])
    data['mass'] = float(buf[4])
    data['deltaeta'] = float(buf[5])
    data['deltaphi'] = float(buf[6])
    data['charge'] = int(buf[7])
    data['ehadovereem'] = float(buf[8])
    data['ncharged'] = int(buf[9])
    data['nneutrals'] = int(buf[10])
    data['tau1'] = float(buf[11])
    data['tau2'] = float(buf[12])
    data['tau3'] = float(buf[13])
    data['tau4'] = float(buf[14])
    data['tau5'] = float(buf[15])
    buf = fin.readline().decode().split() # trimmed/pruned/softdrop P4
    data['pt_trimmed'] = float(buf[0])
    data['eta_trimmed'] = float(buf[1])
    data['phi_trimmed'] = float(buf[2])
    data['mass_trimmed'] = float(buf[3])
    data['pt_pruned'] = float(buf[4])
    data['eta_pruned'] = float(buf[5])
    data['phi_pruned'] = float(buf[6])
    data['mass_pruned'] = float(buf[7])
    data['pt_pruned_sub1'] = float(buf[8])
    data['eta_pruned_sub1'] = float(buf[9])
    data['phi_pruned_sub1'] = float(buf[10])
    data['mass_pruned_sub1'] = float(buf[11])
    data['pt_pruned_sub2'] = float(buf[12])
    data['eta_pruned_sub2'] = float(buf[13])
    data['phi_pruned_sub2'] = float(buf[14])
    data['mass_pruned_sub2'] = float(buf[15])
    data['pt_pruned_sub3'] = float(buf[16])
    data['eta_pruned_sub3'] = float(buf[17])
    data['phi_pruned_sub3'] = float(buf[18])
    data['mass_pruned_sub3'] = float(buf[19])
    data['pt_softdrop'] = float(buf[20])
    data['eta_softdrop'] = float(buf[21])
    data['phi_softdrop'] = float(buf[22])
    data['mass_softdrop'] = float(buf[23])
    data['pt_softdrop_sub1'] = float(buf[24])
    data['eta_softdrop_sub1'] = float(buf[25])
    data['phi_softdrop_sub1'] = float(buf[26])
    data['mass_softdrop_sub1'] = float(buf[27])
    data['pt_softdrop_sub2'] = float(buf[28])
    data['eta_softdrop_sub2'] = float(buf[29])
    data['phi_softdrop_sub2'] = float(buf[30])
    data['mass_softdrop_sub2'] = float(buf[31])
    data['pt_softdrop_sub3'] = float(buf[32])
    data['eta_softdrop_sub3'] = float(buf[33])
    data['phi_softdrop_sub3'] = float(buf[34])
    data['mass_softdrop_sub3'] = float(buf[35])
    buf = fin.readline().decode().split() # subject/constituents counts
    data['nsub_trimmed'] = int(buf[0])
    data['nsub_pruned'] = int(buf[1])
    data['nsub_softdrop'] = int(buf[2])
    data['nconstituents'] = int(buf[3])
    buf = fin.readline().decode().split() # generater info
    data['gen_pid'] = int(buf[0])
    data['gen_charge'] = int(buf[1])
    data['gen_pt'] = float(buf[2])
    data['gen_eta'] = float(buf[3])
    data['gen_phi'] = float(buf[4])
    data['gen_mass'] = float(buf[5])
    
    clist = [] # prepare constituents list
    for i in range(data['nconstituents']):
        var = [float(s) for s in fin.readline().decode().split()] # index, type(0:gen/1:track/2:Ecal/3:Hcal), pid, charge, pt, eta, phi, vx, vy, vz
        # relative to jet pt and direction
        var[_pt ] = var[_pt]/data['pt']
        var[_eta] = var[_eta]-data['eta']
        var[_phi] = deltaPhi(var[_phi],data['phi'])
        clist.append(var)
    clist = np.array(clist)

    buf = fin.readline().decode().split() # Tjet variables, nsub = 1
    data['tjet1_eta1'] = float(buf[0])
    data['tjet1_phi1'] = float(buf[1])
    buf = fin.readline().decode().split()
    data['tjet1_R1'] = float(buf[0])
    data['tjet1_R1_pt1'] = float(buf[1])
    data['tjet1_R1_m1'] = float(buf[2])
    buf = fin.readline().decode().split()
    data['tjet1_R2'] = float(buf[0])
    data['tjet1_R2_pt1'] = float(buf[1])
    data['tjet1_R2_m1'] = float(buf[2])
    buf = fin.readline().decode().split()
    data['tjet1_R3'] = float(buf[0])
    data['tjet1_R3_pt1'] = float(buf[1])
    data['tjet1_R3_m1'] = float(buf[2])
    buf = fin.readline().decode().split()
    data['tjet1_R4'] = float(buf[0])
    data['tjet1_R4_pt1'] = float(buf[1])
    data['tjet1_R4_m1'] = float(buf[2])

    buf = fin.readline().decode().split() # Tjet variables, nsub = 2
    data['tjet2_eta1'] = float(buf[0])
    data['tjet2_phi1'] = float(buf[1])
    buf = fin.readline().decode().split()
    data['tjet2_eta2'] = float(buf[0])
    data['tjet2_phi2'] = float(buf[1])
    buf = fin.readline().decode().split()
    data['tjet2_R1'] = float(buf[0])
    data['tjet2_R1_pt1'] = float(buf[1])
    data['tjet2_R1_pt2'] = float(buf[2])
    data['tjet2_R1_m1'] = float(buf[3])
    data['tjet2_R1_m2'] = float(buf[4])
    buf = fin.readline().decode().split()
    data['tjet2_R2'] = float(buf[0])
    data['tjet2_R2_pt1'] = float(buf[1])
    data['tjet2_R2_pt2'] = float(buf[2])
    data['tjet2_R2_m1'] = float(buf[3])
    data['tjet2_R2_m2'] = float(buf[4])
    buf = fin.readline().decode().split()
    data['tjet2_R3'] = float(buf[0])
    data['tjet2_R3_pt1'] = float(buf[1])
    data['tjet2_R3_pt2'] = float(buf[2])
    data['tjet2_R3_m1'] = float(buf[3])
    data['tjet2_R3_m2'] = float(buf[4])
    buf = fin.readline().decode().split()
    data['tjet2_R4'] = float(buf[0])
    data['tjet2_R4_pt1'] = float(buf[1])
    data['tjet2_R4_pt2'] = float(buf[2])
    data['tjet2_R4_m1'] = float(buf[3])
    data['tjet2_R4_m2'] = float(buf[4])

    buf = fin.readline().decode().split() # Tjet variables, nsub = 3
    data['tjet3_eta1'] = float(buf[0])
    data['tjet3_phi1'] = float(buf[1])
    buf = fin.readline().decode().split()
    data['tjet3_eta2'] = float(buf[0])
    data['tjet3_phi2'] = float(buf[1])
    buf = fin.readline().decode().split()
    data['tjet3_eta3'] = float(buf[0])
    data['tjet3_phi3'] = float(buf[1])
    buf = fin.readline().decode().split()
    data['tjet3_R1'] = float(buf[0])
    data['tjet3_R1_pt1'] = float(buf[1])
    data['tjet3_R1_pt2'] = float(buf[2])
    data['tjet3_R1_pt3'] = float(buf[3])
    data['tjet3_R1_m1'] = float(buf[4])
    data['tjet3_R1_m2'] = float(buf[5])
    data['tjet3_R1_m3'] = float(buf[6])
    buf = fin.readline().decode().split()
    data['tjet3_R2'] = float(buf[0])
    data['tjet3_R2_pt1'] = float(buf[1])
    data['tjet3_R2_pt2'] = float(buf[2])
    data['tjet3_R2_pt3'] = float(buf[3])
    data['tjet3_R2_m1'] = float(buf[4])
    data['tjet3_R2_m2'] = float(buf[5])
    data['tjet3_R2_m3'] = float(buf[6])
    buf = fin.readline().decode().split()
    data['tjet3_R3'] = float(buf[0])
    data['tjet3_R3_pt1'] = float(buf[1])
    data['tjet3_R3_pt2'] = float(buf[2])
    data['tjet3_R3_pt3'] = float(buf[3])
    data['tjet3_R3_m1'] = float(buf[4])
    data['tjet3_R3_m2'] = float(buf[5])
    data['tjet3_R3_m3'] = float(buf[6])
    buf = fin.readline().decode().split()
    data['tjet3_R4'] = float(buf[0])
    data['tjet3_R4_pt1'] = float(buf[1])
    data['tjet3_R4_pt2'] = float(buf[2])
    data['tjet3_R4_pt3'] = float(buf[3])
    data['tjet3_R4_m1'] = float(buf[4])
    data['tjet3_R4_m2'] = float(buf[5])
    data['tjet3_R4_m3'] = float(buf[6])

    if '</jet_data>' not in fin.readline().decode():
        print(">>> ERROR: invalid input", flush=True)
        sys.exit(0)
    
    # Apply rotation
    dir = find_main_axis(clist)
    if dir!=None:
        dir = -dir # rotation everything to x-axis
        clist[:,_eta], clist[:,_phi] = np.cos(dir)*clist[:,_eta]-np.sin(dir)*clist[:,_phi], np.sin(dir)*clist[:,_eta]+np.cos(dir)*clist[:,_phi]

    return data, clist

# skip 1 set of jet data from the gzipped text stream
def skip_jet_data(fin):
    if '<jet_data>' not in fin.readline().decode():
        print(">>> ERROR: invalid input", flush=True)
        sys.exit(0)
    while '</jet_data>' not in fin.readline().decode():
        pass

def prepare_sample(filename, size, skip = 0):
    fin = gzip.open(filename)
    print('Loading from',filename, flush=True)
    
    if skip>0:
        print('Skip first',skip,'jets', flush=True)
        for i in range(skip):
            skip_jet_data(fin)
    
    slist = []
    while(len(slist)<size):
        d, c = parse_jet_data(fin)
        if len(c)<4: continue # drop those jets with low # of constituents
        
        # inject regular jet data for convenience
        supply = np.array([d['pt']/1E3, # in unit of TeV
                           d['eta'],
                           d['phi']],dtype=K.floatx())
        
        c_ext = [] # expend the 2-particle correlations
        for i,j in [(i,j) for i in range(len(c)) for j in range(i+1,len(c))]:
            
            # only (pt,eta,phi)*2
            c_ext.append([c[i,_pt],c[i,_eta],c[i,_phi],
                          c[j,_pt],c[j,_eta],c[j,_phi]])

        slist.append((np.array(c_ext,dtype=K.floatx()),supply)) # tuple of 2pc array, supply array
        if len(slist) % 1000==0: print(len(slist),'/',size,'jets loaded', flush=True)
    fin.close()
    return slist

# custom 2PC layer
class MyLayer(Layer):

    def __init__(self, filters, toppooling_dim = 1, hidden_dim = None, **kwargs):
        self.filters = filters
        self.toppooling_dim = toppooling_dim
        self.hidden_dim = hidden_dim
        super(MyLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        if self.hidden_dim!=None:
            self.w1 = self.add_weight(name='w1', shape=(self.filters, input_shape[-1], self.hidden_dim), initializer='normal', trainable=True)
            self.b1 = self.add_weight(name='b1', shape=(self.filters, self.hidden_dim), initializer='normal', trainable=True)
            self.w2 = self.add_weight(name='w2', shape=(self.filters, self.hidden_dim, 1), initializer='normal', trainable=True)
            self.b2 = self.add_weight(name='b2', shape=(self.filters, 1), initializer='normal', trainable=True)
        else:
            self.w1 = self.add_weight(name='w1', shape=(self.filters, input_shape[-1], 1), initializer='normal', trainable=True)
            self.b1 = self.add_weight(name='b1', shape=(self.filters, 1), initializer='normal', trainable=True)
        
        super(MyLayer, self).build(input_shape)
    
    def call(self, x):
        def operation_over_sample(sample):
            buffer = []
            for idx in range(self.filters):
                oper = K.dot(sample,self.w1[idx])+self.b1[idx] # sum w*x+b
                oper = K.relu(oper) # activation
                
                if self.hidden_dim!=None: # insert hidden dense layer
                    oper = K.dot(oper,self.w2[idx])+self.b2[idx] # sum w*x+b
                    oper = K.relu(oper) # activation
                
                # since the first input is pt, which is always positive => convert it as a mask
                mask = K.expand_dims(K.greater(sample[:,0],0.),axis=1)
                mask = K.cast(mask,K.floatx())
                oper *= mask # apply masking
                
                top_indices = tf.nn.top_k(oper[:,0], k=self.toppooling_dim, sorted=True).indices
                top = K.gather(oper,top_indices) # top-k pooling
                
                buffer.append(top)
            return K.concatenate(buffer)
        return K.map_fn(operation_over_sample,x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.toppooling_dim, self.filters) # output dimension = n_samples x toppooling_dim x filters

slist_sig = prepare_sample(SRC_SIG['filename'], SRC_SIG['n'], SRC_SIG['skip'])
slist_bkg = prepare_sample(SRC_BKG['filename'], SRC_BKG['n'], SRC_BKG['skip'])
sample = [[*s,np.array([1.,0.],dtype=K.floatx())] for s in slist_sig]+[[*s,np.array([0.,1.],dtype=K.floatx())] for s in slist_bkg]
random.shuffle(sample) # shuffle the sample before training/testing

# split training (2/3) and testing (1/3) samples
n_training = len(sample)*2//3
n_testing = len(sample)-n_training
s_train = sample[:n_training]
s_test = sample[n_training:]

# custom layer for 2-particle correlation
x_input = Input(shape=(None,sample[0][0].shape[-1]))
x_seq = MyLayer(filters=64, toppooling_dim=4, hidden_dim=2*sample[0][0].shape[-1])(x_input)
x_seq = Flatten()(x_seq)

# dense layer for supplementary jet data
y_input = Input(shape=(sample[0][1].shape[-1],))
y_seq = Dense(2*sample[0][1].shape[-1], activation='relu')(y_input)

comb_seq = concatenate([x_seq,y_seq])
comb_seq = Dense(128, activation='relu')(comb_seq)
comb_seq = Dense(2, activation='softmax')(comb_seq)
model = Model([x_input,y_input], comb_seq)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
print(model.summary(), flush=True)

patience, best_los = MAX_PATIENCE, 1E10
for epoch in range(EPOCHS): # user training loop for dynamic memory allocation

    print('Epoch: %d/%d' % (epoch+1,EPOCHS), flush=True)
    random.shuffle(s_train)
    
    train_ret = []
    for batch in range(n_training//BATCH_SIZE):
    
        # prepare taining sample, padding the 2PC inputs into the same shape within the batch
        b_train = s_train[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
        max_perm = max([len(x) for x,y,z in b_train])
        x_train = np.array([np.pad(x,((0,max_perm-len(x)),(0,0)),mode='constant',constant_values=0.) for x,y,z in b_train])
        y_train = np.array([y for x,y,z in b_train])
        z_train = np.array([z for x,y,z in b_train])
        
        ret = model.train_on_batch([x_train,y_train], z_train)
        train_ret.append(ret)
        train_los = np.array(train_ret)[:,0].mean()
        train_acc = np.array(train_ret)[:,1].mean()
        print('\rbatch: %05d/%05d, loss: %.4f, acc: %.4f' % (batch+1,n_training//BATCH_SIZE,train_los,train_acc),end='', flush=True)
        
    test_ret = []
    for batch in range(n_testing//BATCH_SIZE):
    
        # prepare testing sample
        b_test = s_test[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
        max_perm = max([len(x) for x,y,z in b_test])
        x_test = np.array([np.pad(x,((0,max_perm-len(x)),(0,0)),mode='constant',constant_values=0.) for x,y,z in b_test])
        y_test = np.array([y for x,y,z in b_test])
        z_test = np.array([z for x,y,z in b_test])
            
        ret = model.test_on_batch([x_test,y_test], z_test)
        test_ret.append(ret)
        
    test_los = np.array(test_ret)[:,0].mean()
    test_acc = np.array(test_ret)[:,1].mean()
    print(', val_loss: %.4f, val_acc: %.4f' % (test_los,test_acc), flush=True)
    
    # early stopping by test_los from testing sample
    # one has to measure the actual performance from another independent sample!
    if test_los<best_los:
        best_los = test_los
        model.save_weights(WEIGHTS_2PCNN)
        patience = MAX_PATIENCE
    else:
        patience -= 1
        if patience<=0:
            print('Stopped; no improvement after %d epochs.' % MAX_PATIENCE, flush=True)
            break
