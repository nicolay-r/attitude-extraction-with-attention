#!/usr/bin/env bash

pushd .
cd ctx
python -u att_bilstm.py > __att-bilstm.txt
python -u att_ef_bilstm.py > __att-ef-bilstm.txt
python -u att_se_bilstm.py > __att-se-bilstm.txt
python -u att_sef_bilstm.py > __att-sef-bilstm.txt
python -u att_frames_bilstm.py > __att-frames-bilstm.txt
python -u att_hidden_z_yang.py > __att-hidden-zyang.txt
python -u bilstm.py > __bilstm-log.txt
python -u ian_ef.py > __ian-ef.txt
python -u ian_se.py > __ian-se.txt
python -u ian_sef.py > __ian-sef.txt
python -u ian_ends.py > __ian-ends.txt
python -u ian_frames.py > __ian-frames.txt
python -u lstm.py > __lstm-log.txt
python -u rcnn.py > __rcnn-log.txt
python -u self_att_bilstm.py > __self-att-bilstm-log.txt
popd

pushd .
cd mi
python -u att_bilstm.py > __att-bilstm.txt
python -u att_ef_bilstm.py > __att-ef-bilstm.txt
python -u att_se_bilstm.py > __att-se-bilstm.txt
python -u att_sef_bilstm.py > __att-sef-bilstm.txt
python -u att_frames_bilstm.py > __att-frames-bilstm.txt
python -u att_hidden_z_yang.py > __att-hidden-zyang.txt
python -u bilstm.py > __bilstm-log.txt
python -u ian_ef.py > __ian-ef.txt
python -u ian_se.py > __ian-se.txt
python -u ian_sef.py > __ian-sef.txt
python -u ian_ends.py > __ian-ends.txt
python -u ian_frames.py > __ian-frames.txt
python -u lstm.py > __lstm-log.txt
python -u rcnn.py > __rcnn-log.txt
python -u self_att_bilstm.py > __self-att-bilstm-log.txt
popd

pushd .
cd mi_att
python -u att_bilstm.py > __att-bilstm.txt
python -u att_ef_bilstm.py > __att-ef-bilstm.txt
python -u att_se_bilstm.py > __att-se-bilstm.txt
python -u att_sef_bilstm.py > __att-sef-bilstm.txt
python -u att_frames_bilstm.py > __att-frames-bilstm.txt
python -u att_hidden_z_yang.py > __att-hidden-zyang.txt
python -u bilstm.py > __bilstm-log.txt
python -u ian_ef.py > __ian-ef.txt
python -u ian_se.py > __ian-se.txt
python -u ian_sef.py > __ian-sef.txt
python -u ian_ends.py > __ian-ends.txt
python -u ian_frames.py > __ian-frames.txt
python -u lstm.py > __lstm-log.txt
python -u rcnn.py > __rcnn-log.txt
python -u self_att_bilstm.py > __self-att-bilstm-log.txt
popd

