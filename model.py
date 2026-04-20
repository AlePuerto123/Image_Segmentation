import segmentation_models_pytorch as smp
import config

def get_model():

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=config.NUM_CLASSES,
    )

    return model.to(config.DEVICE)