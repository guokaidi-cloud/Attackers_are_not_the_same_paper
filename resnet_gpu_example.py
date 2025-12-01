#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet GPU使用示例

演示如何在GPU上使用ResNet模型
"""

import torch
from utils.models import ResNet, ResidualBlock

def test_resnet_gpu():
    """测试ResNet在GPU上的运行"""
    print("🧪 测试ResNet GPU功能...")
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ CUDA可用! GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("⚠️  CUDA不可用，使用CPU")
    
    # 创建ResNet模型 (CIFAR-10)
    model = ResNet(
        block=ResidualBlock,
        layers=[2, 2, 2, 2],
        num_classes=10,
        num_passive=4,
        padding_mode=False,
        division_mode='vertical',
        device=device
    )
    
    print(f"📍 模型设备: {model.get_device()}")
    
    # 创建示例数据 (4个passive client的数据)
    batch_size = 16
    sample_data = []
    for i in range(4):
        # 每个客户端有部分CIFAR-10数据 (垂直分割)
        sample_data.append(torch.randn(batch_size, 3, 32, 8).to(device))
    
    # 前向传播测试
    try:
        with torch.no_grad():
            embeddings, logits, predictions = model(sample_data)
        
        print(f"✅ 前向传播成功!")
        print(f"📊 输出形状 - Logits: {logits.shape}, Predictions: {predictions.shape}")
        print(f"📈 预测概率范围: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
        
        if device.type == 'cuda':
            print(f"🎯 输出在GPU: {logits.is_cuda}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 ResNet GPU支持测试")
    print("=" * 30)
    
    success = test_resnet_gpu()
    
    if success:
        print("\n🎉 ResNet GPU支持正常!")
        print("\n💡 使用方法:")
        print("# 创建GPU上的ResNet模型")
        print("device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        print("model = ResNet(..., device=device)")
        print("# 或者手动设置设备")
        print("model.set_device(torch.device('cuda'))")
    else:
        print("\n❌ 测试失败")

if __name__ == "__main__":
    main()
