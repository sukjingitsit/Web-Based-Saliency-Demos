class SessionState:
    def __init__(self):
        self.run = True
        self.model = "ResNet50V2"
        self.method = "Integrated Gradients"
        self.smoothgrad =  False
        self.idgi = False
        self.image = False
        self.classchoice = "Top Class"
        self.classnum = -1
        self.steps = 20
        self.baseline = "black",
        self.max_sig = 0.5
        self.grad_step = 0.01
        self.sqrt = False
        self.noise_steps = 20
        self.noise_var = 0.1
        self.steps_at = "-"
        self.image_arr = None
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)