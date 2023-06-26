from .model import BaseModel 
from .be_your_own_teacher import BeYourOwnTeacher

def load_model(model_name, base_model, pretrained, num_class):
    model = __import__('models').__dict__[model_name](  model_name = base_model,
                                                                pretrained = pretrained,
                                                                num_class  = num_class)
    return model 