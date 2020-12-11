CATEGORIES = {}

def get_name_color(source:str, category_number:int):
    try:
        infos = CATEGORIES[source][str(category_number)]
        return infos['name'], infos['color']
    except:
        return '?',(128,128,128)

def get_category_number(source:str, name:str):
    for number in CATEGORIES[source]:
        if CATEGORIES[source][number]['name'] == name:
            return int(number)
    raise ValueError(f"Could not find category {name} in {source}'s categories.")

def get_source(ds_type):
    parts = ds_type.split('-')
    for part in parts:
        for source in CATEGORIES:
            if part.lower() == source.lower():
                return part
    return 'deepen'


CATEGORIES['deepen'] = {
    '0':{'name':'pedestrian','color':(242, 132, 48)},
    '1':{'name':'deformed pedestrian','color':(120, 152, 48)},
    '2':{'name':'bicycle','color':(48, 242, 183)},
    '3':{'name':'car','color':(241, 48, 242)},
    '4':{'name':'van','color':(185, 55, 230)},
    '5':{'name':'bus','color':(242, 48, 74)},
    '6':{'name':'truck','color':(138, 48, 242)},
    '7':{'name':'motorcycle','color':(183, 242, 48)},
    '8':{'name':'stop sign','color':(119, 242, 48)},
    '9':{'name':'traffic light','color':(48, 235, 242)},
    '10':{'name':'traffic sign','color':(48, 25, 212)},
    '11':{'name':'traffic cone','color':(230, 110, 60)},
    '12':{'name':'fire hydrant','color':(242, 48, 177)},
    '13':{'name':'guard rail','color':(140, 120, 130)},
    '14':{'name':'pole','color':(80, 120, 130)},
    '15':{'name':'pole group','color':(80, 120, 180)},
    '16':{'name':'road','color':(30, 30, 30)},
    '17':{'name':'sidewalk','color':(30, 70, 30)},
    '18':{'name':'wall','color':(230, 230, 130)},
    '19':{'name':'building','color':(230, 130, 130)},
    '20':{'name':'vegetation','color':(60, 250, 60)},
    '21':{'name':'terrain','color':(100, 140, 40)},
    '22':{'name':'ground','color':(100, 40, 40)},
    '23':{'name':'crosstalk','color':(250, 10, 10)},
    '24':{'name':'noise','color':(250, 250, 250)},
    '25':{'name':'others','color':(128, 128, 128)},
    '26':{'name':'animal','color':(250, 250, 10)},
    '27':{'name':'unpainted','color':(255, 255, 255)},
    '28':{'name':'cyclist', 'color':(198, 238, 242)},
    '29':{'name':'motorcyclist', 'color':(100, 152, 255)},
    '30':{'name':'unclassified vehicle', 'color':(50, 130, 200)},
    '31':{'name':'obstacle', 'color':(100, 200, 50)},
    '32':{'name':'trailer', 'color':(255, 150, 120)},
    '33':{'name':'barrier', 'color':(100, 190, 240)},
    '34':{'name':'bicycle rack', 'color':(20, 90, 200)},
    '35':{'name':'construction vehicle', 'color':(80, 40,0)}
}

CATEGORIES['maskrcnn'] = {
    '0':{'name':'background','color':(255, 255, 255)},
    '1':{'name':'person','color':(242, 132, 48)},
    '2':{'name':'bicycle','color':(48, 242, 183)},
    '3':{'name':'car','color':(241, 48, 242)},
    '4':{'name':'motorcycle','color':(183, 242, 48)},
    '5':{'name':'airplane','color':(48, 125, 242)},
    '6':{'name':'bus','color':(242, 48, 74)},
    '7':{'name':'train','color':(48, 242, 80)},
    '8':{'name':'truck','color':(138, 48, 242)},
    '9':{'name':'boat','color':(242, 196, 48)},
    '10':{'name':'traffic light','color':(48, 235, 242)},
    '11':{'name':'fire hydrant','color':(242, 48, 177)},
    '12':{'name':'stop sign','color':(119, 242, 48)},
    '13':{'name':'parking meter','color':(48, 61, 242)},
    '14':{'name':'bench','color':(242, 87, 48)},
    '15':{'name':'bird','color':(48, 242, 145)},
    '16':{'name':'cat','color':(203, 48, 242)},
    '17':{'name':'dog','color':(222, 242, 48)},
    '18':{'name':'horse','color':(48, 171, 242)},
    '19':{'name':'sheep','color':(242, 48, 112)},
    '20':{'name':'cow','color':(54, 242, 48)},
    '21':{'name':'elephant','color':(100, 48, 242)},
    '22':{'name':'bear','color':(242, 151, 48)},
    '23':{'name':'zebra','color':(48, 242, 209)},
    '24':{'name':'giraffe','color':(242, 48, 216)},
    '25':{'name':'backpack','color':(158, 242, 48)},
    '26':{'name':'umbrella','color':(48, 106, 242)},
    '27':{'name':'handbag','color':(242, 48, 48)},
    '28':{'name':'tie','color':(48, 242, 106)},
    '29':{'name':'suitcase','color':(164, 48, 242)},
    '30':{'name':'frisbee','color':(242, 216, 48)},
    '31':{'name':'skis','color':(48, 209, 242)},
    '32':{'name':'snowboard','color':(242, 48, 151)},
    '33':{'name':'sports ball','color':(93, 242, 48)},
    '34':{'name':'kite','color':(54, 48, 242)},
    '35':{'name':'baseball bat','color':(242, 112, 48)},
    '36':{'name':'baseball glove','color':(48, 242, 171)},
    '37':{'name':'skateboard','color':(229, 48, 242)},
    '38':{'name':'surfboard','color':(203, 242, 48)},
    '39':{'name':'tennis racket','color':(48, 145, 242)},
    '40':{'name':'bottle','color':(242, 48, 87)},
    '41':{'name':'wine glass','color':(48, 242, 67)},
    '42':{'name':'cup','color':(125, 48, 242)},
    '43':{'name':'fork','color':(242, 177, 48)},
    '44':{'name':'knife','color':(48, 242, 235)},
    '45':{'name':'spoon','color':(242, 48, 190)},
    '46':{'name':'bowl','color':(132, 242, 48)},
    '47':{'name':'banana','color':(48, 80, 242)},
    '48':{'name':'apple','color':(242, 74, 48)},
    '49':{'name':'sandwich','color':(48, 242, 132)},
    '50':{'name':'orange','color':(190, 48, 242)},
    '51':{'name':'broccoli','color':(242, 241, 48)},
    '52':{'name':'carrot','color':(48, 183, 242)},
    '53':{'name':'hot dog','color':(242, 48, 125)},
    '54':{'name':'pizza','color':(67, 242, 48)},
    '55':{'name':'donut','color':(80, 48, 242)},
    '56':{'name':'cake','color':(242, 138, 48)},
    '57':{'name':'chair','color':(48, 242, 196)},
    '58':{'name':'couch','color':(242, 48, 229)},
    '59':{'name':'potted plant','color':(177, 242, 48)},
    '60':{'name':'bed','color':(48, 119, 242)},
    '61':{'name':'dining table','color':(242, 48, 61)},
    '62':{'name':'toilet','color':(48, 242, 93)},
    '63':{'name':'tv','color':(145, 48, 242)},
    '64':{'name':'laptop','color':(242, 203, 48)},
    '65':{'name':'mouse','color':(48, 222, 242)},
    '66':{'name':'remote','color':(242, 48, 164)},
    '67':{'name':'keyboard','color':(112, 242, 48)},
    '68':{'name':'cell phone','color':(48, 54, 242)},
    '69':{'name':'microwave','color':(242, 100, 48)},
    '70':{'name':'oven','color':(48, 242, 158)},
    '71':{'name':'toaster','color':(209, 48, 242)},
    '72':{'name':'sink','color':(216, 242, 48)},
    '73':{'name':'refrigerator','color':(48, 158, 242)},
    '74':{'name':'book','color':(242, 48, 100)},
    '75':{'name':'clock','color':(48, 242, 48)},
    '76':{'name':'vase','color':(106, 48, 242)},
    '77':{'name':'scissors','color':(242, 164, 48)},
    '78':{'name':'teddy bear','color':(48, 242, 222)},
    '79':{'name':'hair drier','color':(242, 48, 209)},
    '80':{'name':'toothbrush','color':(151, 242, 48)},
}

CATEGORIES['detectron'] = CATEGORIES['maskrcnn']
CATEGORIES['detectron']['81'] = {'name':'road','color':(168, 130, 180)}

CATEGORIES['toynet'] = {
    '0':{'name':'pedestrian','color':(242, 132, 48)},
    '1':{'name':'vehicle','color':(120, 48, 242)},
}

CATEGORIES['carla'] = {
    '0':{'name':'none','color':(0, 0, 0)},
    '1':{'name':'building','color':(70, 70, 70)},  
    '2':{'name':'fences','color':(190, 153, 153)}, 
    '3':{'name':'other','color':(72, 0, 90)},      
    '4':{'name':'pedestrian','color':(220, 20, 60)}, 
    '5':{'name':'pole','color':(153, 153, 153)},   
    '6':{'name':'roadline','color':(157, 234, 50)}, 
    '7':{'name':'road','color':(128, 64, 128)},   
    '8':{'name':'sidewalk','color':(244, 35, 232)},   
    '9':{'name':'vegetation','color':(107, 142, 35)}, 
    '10':{'name':'vehicle','color':(0, 0, 255)},     
    '11':{'name':'wall','color':(102, 102, 156)},  
    '12':{'name':'traffic sign','color':(220, 220, 0)},
    '13':{'name':'sky','color':(70, 130, 180)},
    '14':{'name':'ground','color':(81, 0, 81)},
    '15':{'name':'bridge','color':(150, 100, 100)},
    '16':{'name':'rail track','color':(230, 150, 140)},
    '17':{'name':'guard rail','color':(180, 165, 180)},
    '18':{'name':'traffic light','color':(250, 170, 30)},
    '19':{'name':'static','color':(110, 190, 160)},
    '20':{'name':'dynamic','color':(170, 120, 50)},
    '21':{'name':'water','color':(45, 60, 150)},
    '22':{'name':'terrain','color':(145, 170, 100)},
    '23':{'name':'bike','color':(198, 238, 242)},
}


