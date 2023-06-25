from .model import * 

def load_model(model_name,pretrained,num_class):
    model = BaseModel(
        model_name = model_name,
        num_class  = num_class,
        pretrained = pretrained
    )
    return model 