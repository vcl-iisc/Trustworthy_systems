mode = ['trust_more_DeepFool_0_robust_resnet18_train', 'ddbs_cdb_DeepFool_0_robust_resnet18_train']
dataset = 'cifar10'
model = 'MobileNetV2'
per_cls = 100
teacher_model = 'ResNet18'
t_adv = 'adv'
alpha = 0.8
temp = 8.0

student_load_path = f'_code/checkpoint/checkpoint/{dataset}/{model}_{mode[1]}_{per_cls}_teacher_{teacher_model}_{t_adv}_False_{alpha}_{temp}.t7'

print(student_load_path)
# print('./checkpoint/cifar10/MobileNetV2_trust_more_DeepFool_0_robust_resnet18_train_100_teacher_ResNet18_adv_False_0.8_8.0.t7')