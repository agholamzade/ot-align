from ml_collections import ConfigDict

def get_config():

    config = ConfigDict()
    
    config.vf_small = ConfigDict({"hidden_dims": [128]*8, "time_dims": [128]*8, "condition_dims": [128]*8, "output_dims": 8*[256]}) #1.2M
    config.vf_medium = ConfigDict({"hidden_dims": [256]*8, "time_dims": [256]*8, "condition_dims": [256]*8, "output_dims": 6*[1024]})#8.1ßM
    config.vf_large = ConfigDict({"hidden_dims": [1024]*8, "time_dims": [1024]*8, "condition_dims": [1024]*8, "output_dims": 8*[1024]})#59M

    config.adanl_small = ConfigDict({"hidden_dims": [128]*2, "time_dims": [128]*2, "condition_dims": [128]*2, "output_dims": 6*[256]})#1.2M
    config.adanl_medium = ConfigDict({"hidden_dims": [384]*2, "time_dims": [384]*2, "condition_dims": [384]*2, "output_dims": 8*[384]})#8M
    config.adanl_large = ConfigDict({"hidden_dims": [1024]*2, "time_dims": [1024]*2, "condition_dims": [1024]*2, "output_dims": 8*[1024]})#57M
