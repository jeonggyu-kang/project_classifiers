from .mymodel import CustomizationNet 

def get_model(model_name, n_classes):
    model = CustomizationNet( n_classes, model_name, (2,3 , 224, 224))

    return model