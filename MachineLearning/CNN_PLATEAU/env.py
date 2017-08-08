import configparser

class Env(object):
    """
    Enviroment of this model
    Change occurs only in main function

    Instance
    ________
    config_var : dict
        dictionary of configuration variable
    file : dict
        file path information
    """
    def __init__(self):
        self.config={}
        self.file={}
        
    def load_config(self,config_path):
        """
        load environment variable from config.ini

        Parameter
        _________
        config_path : string
            configuration file path 
        """
        config = configparser.ConfigParser()
        config.read(config_path)

        # store to env.config
        for section in config.sections():
            self.config[section]={}
            for option in config.options(section):
               self.config[section][option]=config.get(section,option)

    def get_config(self,section,option,type="str"):
        result=self.config[section][option]
        # change type
        if type=="int":
            return int(result)
        elif type=="float":
            return float(result)
        elif type=="list":
            return _listify(result,",")
        else:
            return result


def _listify(var,spliter):
    result=var.split(spliter)
    return result

def int_input(message=""):
    """
    transform user input into int format
    if user input is not valid(=input can't be interpreted as int), ask again

    Parameter
    _________
    message : string, option
        print explaining message 
    
    Return
    ______
    val : int
        int type of user input
    """
    if len(message)>0:
        print(message)
    while True:
        val=input()
        try:
            val=int(val)
        except:
            print("Wrong input")
            continue
        break
    return val


