```
python dataset/inspect_tfrecords.py \
--file_pattern $HOME/data/pattern_detection_tokam/tfrecords/blob_i.tfrecord \
--model_name "efficientdet-d0" \
--samples 10 \
--save_samples_dir train_samples/ \
--hparams="label_map={1:'label1'}"
```