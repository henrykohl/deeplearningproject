import io

import bentoml
import numpy as np
import torch
from bentoml.io import Image, Text
from PIL import Image as PILImage

from Xray.constant.training_pipeline import *

bento_model = bentoml.pytorch.get(BENTOML_MODEL_NAME) # returns A Model
## his retrieves a saved PyTorch model from the BentoML model store, identified by its name and tag.

runner = bento_model.to_runner()
## transforms the retrieved model into a Runner. Runners are objects that encapsulate the model 
# and provide a standardized way to interact with it, such as for prediction.

svc = bentoml.Service(name=BENTOML_SERVICE_NAME, runners=[runner])
## for building a BentoML service. It defines how your model will be exposed as an API. 
# Use bentoml.Service to define the input/output interfaces, the runners to use, and the overall service logic.

@svc.api(input=Image(allowed_mime_types=["image/jpeg"]), output=Text())
async def predict(img): # img 應該是 PIL.JpegImagePlugin.JpegImageFile 類型
    b = io.BytesIO() # _io.BytesIO 類型

    img.save(b, "jpeg") # 把 img 存到 BytesIO

    im_bytes = b.getvalue() # im_bytes 是 bytes 類型

    my_transforms = bento_model.custom_objects.get(TRAIN_TRANSFORMS_KEY) ## 從 bento model 提取 轉型模式物件 

    image = PILImage.open(io.BytesIO(im_bytes)).convert("RGB") # image 是 PIL.Image.Image 類型

    image = torch.from_numpy(np.array(my_transforms(image).unsqueeze(0))) # my_transforms(image) 是 torch.Tensor 類型
    # torch.from_numpy(np.array()) 似乎多此一舉
    # my_transforms(image).unsqueeze(0) 的大小是 [1,3,224,224]
    # image 的大小也是 [1,3,224,224]

    image = image.reshape(1, 3, 224, 224) # torch.tensor 類型

    batch_ret = await runner.async_run(image) # (應該也是) torch.tensor 類型，大小為  [1,2]

    pred = PREDICTION_LABEL[max(torch.argmax(batch_ret, dim=1).detach().cpu().tolist())] # tolist 轉 tensor 為 list

    return pred # 結果為 字串