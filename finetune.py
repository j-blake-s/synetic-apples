import ultralytics


'''

NOTE:
adjust this code as needed - the below are numbers for a 8xB200 system


'''
dir = f'./'

if __name__ == "__main__":
  ultralytics.checks()

  epochs = 100

  dataNames = [
    # 'finetune',
    # 'real',
    'synetic+bg-train+real-val',
  ]

  hyperparams = [
    ('12', 'n', dir),
  ]


  savePeriod = epochs//10
  if savePeriod <= 0:
    savePeriod = 1

  for hyperparam, dataName in zip(hyperparams, dataNames):
    modelVersion, modelSize, pathDataYaml = hyperparam
    
    pathDataYaml = f'{pathDataYaml}/{dataName}.yaml'

    projectName = f'ApplesM5_{modelVersion}{modelSize}'
    taskName = 'detect'

    modelName = f"yolo{modelVersion}{modelSize}.yaml"

    trainName = f'{projectName}-{taskName}-{epochs}_{dataName}_0'

    # modelDet = ultralytics.YOLO(modelName)
    modelDet = ultralytics.YOLO(f'/home/blake/Desktop/github/synetic-apples/runs/detect/ApplesM5_12n-detect-100_synetic+bg-train+real-val_010/weights/best.pt')

    devices = [1]
    devicesLen = len(devices)

    batchSize = devicesLen * 8 * 2

    batchSize = int(batchSize)

    results = modelDet.train(
      imgsz=640,

      name=trainName,
      data=pathDataYaml,
      task=taskName,
      epochs=epochs,
      device=devices,
      batch=batchSize,
      workers=28,

      cache='disk',

      # freeze=21,
      flipud=0.5,
      fliplr=0.5,

      hsv_h=0.1,
      hsv_s=0.1,
      hsv_v=0.1,

      mosaic=0.75,
      close_mosaic=0,

      degrees=45.0,
      shear=15.0,
      perspective=0.0005,
      translate=0.3,
      mixup=0.1,        # image mixup (probability)
      copy_paste=0.1,   # segment copy-paste (probability)
      auto_augment='randaugment', # (str) auto augmentation policy for classification (randaugment, autoaugment, augmix)
      augment=True,

      val=True,

    )

