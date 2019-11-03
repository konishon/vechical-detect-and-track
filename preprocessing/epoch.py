import re
pattern = re.compile("id \w* \(")

numbers = {0}

for i, line in enumerate(open('/home/nishon/Projects/python/vechical-detect-and-track/data/raw/code')):
    for match in re.finditer(pattern, line):
        value = match.group().split(" ")[1]
        numbers.add(value)
print(numbers)        

!python vechical-detect-and-track-1-feature-deep-sort-tracker/aerial_pedestrian_detection/keras_retinanet/bin/train.py --weights '/content/drive/My Drive/Traffic Flow/Colab/Day three/mobilenet128_csv_02.h5' --backbone mobilenet128 --freeze-backbone --epochs 10  --snapshot-path '/content/drive/My Drive/Traffic Flow/Colab' csv train.csv vechical-detect-and-track-1-feature-deep-sort-tracker/data/labels.csv --val-annotations  vechical-detect-and-track-1-feature-deep-sort-tracker/data/val.csv