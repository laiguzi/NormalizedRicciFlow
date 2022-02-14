
#!/usr/bin/env bash
config=config_SBM_0.45.json
main_fun='main_surgery_gamma.py'
#model_name='StarNormalize'
#python $main_fun $config $model_name

#model_name='StarUnnormalize'
#python $main_fun $config $model_name

model_name='aStarNormalize'
python $main_fun $config $model_name


#model_name='OllivierNormalize'
#python $main_fun $config $model_name

model_name='OllivierUnnormalize'
python $main_fun $config $model_name

model_name='aOllivierNormalize'
python $main_fun $config $model_name
