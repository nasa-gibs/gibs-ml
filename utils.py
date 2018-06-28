import subprocess

def run_command(cmd):
    """
    Runs the provided command on the terminal.
    Arguments:
        cmd -- the command to be executed.
    """
    print(' '.join(cmd))
    process = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    process.wait()
    for output in process.stdout:
        if b"ERROR" in output:
            raise Exception(error.strip())
    for error in process.stderr:
        raise Exception(error.strip())
    
def get_last_updated(metadata_json):
    try:
        json_file = open(metadata_json, 'r')
    except IOError:
        return None, None
    
    metadata = json.load(json_file)
    json_file.close()
    return metadata.get("updated", None), metadata.get("date", None)

def get_old_layer_info(metadata_json, layer_name):
    try:
        json_file = open(metadata_json, 'r')
    except IOError:
        return None
    old_layer = None
    metadata = json.load(json_file)
    json_file.close()
    layers = metadata.get("layers", [])
    for layer in layers:
        if layer.get("name", "") == layer_name:
            old_layer = GIBSLayer(layer.get("title", ""), layer.get("source", ""), layer.get("uom", ""), layer.get("min", ""), layer.get("max", ""), layer.get("name", ""), metadata.get("epsg", ""), "", "2km", metadata.get("land", 0), "", layer.get("date", None))
    return old_layer

def add_error_msg(error_msg, error_msg_list):
    if error_msg not in error_msg_list and "Could not find data for MODIS_Aqua_Cloud_Effective_Radius" not in error_msg:
        error_msg_list.append(error_msg)
        print(error_msg)