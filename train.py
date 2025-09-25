import ultralytics

'''

NOTE:
adjust this code as needed - the below are numbers for a 8xB200 system


'''
def train_model(args, model, trainName, taskName, devices, batchSize):
  results = model.train(
    imgsz=640,
    name=trainName,
    data='data.yaml',
    task=taskName,
    epochs=args.epochs,
    device=devices,
    batch=batchSize,
    workers=28,

    cache='disk',

    freeze=args.freeze,
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

  return results

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data', default=None, type=str, help='path to data directory')
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
    parser.add_argument('-b','--batch_size', default=16, type=int, help='batch_size')
    parser.add_argument('--freeze', default=0, type=int, help='number of layers to freeze')
    parser.add_argument('--checkpoint', default=None, type=str, help="path to model directory")

    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
  ultralytics.checks()

  args = parse_args()
  if args.data[-1] == "/": args.data = args.data[:-1]
  if args.checkpoint is not None and args.checkpoint[-1] == "/": args.checkpoint = args.checkpoint[:-1]

  # Save Period
  savePeriod = args.epochs // 10
  if savePeriod <= 0:
    savePeriod = 1
  
  # Devices
  devices = [0]
  devicesLen = len(devices)
  batchSize = int(devicesLen * args.batch_size)

  # File Names
  projectName = f'apples'
  taskName = 'detect'
  dataName = args.data.split("/")[-1]
  if args.checkpoint is not None:
    oldProjectName = args.checkpoint.split('/')[-1]
    trainName = f'{oldProjectName}_{dataName}_0'
  else:
    trainName = f'{projectName}-{args.epochs}_{dataName}_0'
  modelVersion = '12'
  modelSize = 'n'
  modelName = f"yolo{modelVersion}{modelSize}.yaml"

  # Model
  if args.checkpoint is not None:
    modelDet = ultralytics.YOLO(f'{args.checkpoint}/weights/best.pt')
  else:
    modelDet = ultralytics.YOLO(modelName)

  # Write Data Yaml
  with open("./data.yaml",'w') as file:
    file.write(f'train: {args.data}/yolo/images/trains\n')
    file.write(f'val: {args.data}/yolo/images/vals\n')
    file.write(f'nc: 1\n')
    file.write(f'names:\n')
    file.write(f'  0: \'apple\'\n')
  
  # Train
  train_model(args, modelDet,  trainName, taskName, devices, batchSize)

