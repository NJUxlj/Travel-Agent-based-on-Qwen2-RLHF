import torch

import torch.nn as nn

import onnx
import onnxruntime as ort

class TorchModel(nn.Module):
    def __init__(self, ):
        super().__init__()
        
        
        self.layer = nn.Linear(100,100)
        
        
        # print("self.layer.shape = ", self.layer.size())
    
    
    
    def forward(self, x  , attention_mask=None, labels=None):
        
        print("x.shape = ", x.shape)
        output = self.layer(x)
        
        
        return output
    
    
    
    
    



if __name__ == '__main__':
    model = TorchModel()
    
    
    params = {}
    # model._save_to_state_dict(params)
    
    # torch.save(model.state_dict(), "model.pt")
    
    # torch.save()
    
    
    # torch.onnx.
    
    
    
    # state_dict = torch.load("model.pt")
    # model.load_state_dict(state_dict)
    
    
    
    # inputs = torch.randn(1, 100)
    
    # print("inputs.shape", inputs.shape)
    
    # torch.onnx.export(
    #     model,
    #     inputs,
    #     "model.onnx",
    #     input_names = ['x'],
    #     output_names = ['output'],
    #     dynamic_axes={
    #         "x":{0: 'batch_size'},
    #         "output":{0: 'batch_size'}
    #     }
    # )
    
    
    # 创建ONNX Runtime推理会话
    ort_session = ort.InferenceSession("model.onnx")
    
    # 准备输入数据（需要与导出时的形状一致）
    test_input = torch.randn(1, 100).numpy()  # ONNX Runtime需要numpy数组
    
    # 运行推理
    outputs = ort_session.run(
        None,  # 输出节点名称列表，None表示获取所有输出
        {'x': test_input}  # 输入字典（键名需与导出时input_names一致）
    )
    
    
    print("outputs = ", outputs)
    
    