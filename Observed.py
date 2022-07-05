def getSysDict(sysname):
    Sys_Dict ={
        'J1305':{
            'ra':196.73033374444574,
            'dec':-70.4514169682461,
            'd': [7.5,1.8,1.4], #med, plus, minus
            'pm_ra': [-7.8887141570421475, 0.6188476],
            'pm_dec': [-0.15672329419678083,0.7175815],
            'v_rad': [-9., 5.],
            'd_skew': [1.6218479298914317, 6.178163635584847, 2.183950249786391],
            'fullname': 'MAXI J1305-704'
            },

        'GRS1915': {
            'ra': 288.7981250,
            'dec' : 10.9457667,
            'd' : [8.6,2.0, 1.6],
            'pm_ra' : [-3.19,0.03],#actually pm_ra*cos(dec)
            'pm_dec':[-6.24,0.05],
            'v_rad':[11.0,4.5],#[12.3,1.0],
            
            'fullname': 'GRS 1915+105'

        },
        
        'V404': {
            'ra': 306.0158333,
            'dec': 33.8675556,
            'd': [2.39,0.14],
            'pm_ra': [-5.04,0.22],
            'pm_dec': [-7.64,0.03],
            'v_rad': [-0.4,2.2],

            'fullname': 'V404 Cygni'
        },
        
        'SCO': {
            'l': 359.09,
            'b': 23.78,
            'd': [2.8,0.3],
            'pm_ra': [-6.88,0.07],
            'pm_dec': [-12.02,0.16],
            'v_rad': [-113.8,0.6],
            
            'fullname': 'SCO X-1'
        },
        
        'LS': {
            'l': 16.88,
            'b': -1.29,
            'd': [2.9,0.3],
            'pm_ra': [7.10,0.13],
            'pm_dec': [-8.75,0.16],
            'v_rad': [17.2,0.5],
            
            'fullname': 'LS 5039'
        },
        
        'Aql': {
            'l': 34.67,
            'b': -4.68,
            'd': [5.0,0.9],
            'pm_ra': [-2.64,0.14],
            'pm_dec': [-3.53,1.40],
            'v_rad': [30,10],
            
            'fullname': 'Aql X-1'
        },
        
        'Cyg': {
            'l': 87.33,
            'b': -11.32,
            'd': [11,2],
            'pm_ra': [-3.00,0.68],
            'pm_dec': [-0.64,0.68],
            'v_rad': [-209.6,0.8],
            
            'fullname': 'Cyg X-2'
        },
        
        'XTEJ1118':{
            'l': 157.66,
            'b': 62.32,
            'd': [1.72, 0.10],
            'pm_ra': [-16.8, 1.6],
            'pm_dec': [-7.4, 1.6],
            'v_rad': [2.7, 1.1],
            
            'fullname': 'XTE J1118+480'
        },
        
        'GROJ1655':{
            'l': 344.98,
            'b': 2.46,
            'd': [3.2, 0.2],
            'pm_ra': [-3.3, 0.50],
            'pm_dec': [-4.0, 0.4],
            'v_rad': [-141.9, 1.3],
            
            'fullname': 'GRO J1655-40'
        },
        
        'LSI61':{
            'l': 135.68,
            'b': 1.09,
            'd': [2.0, 0.2],
            'pm_ra': [-0.30, 0.07],
            'pm_dec': [-0.26, 0.05],
            'v_rad': [-40.2, 1.9],
            
            'fullname': 'LSI +61 303'
        },
        
        'CEN':{
            'l': 332.24,
            'b': 23.88,
            'd': [1.4, 0.3],
            'pm_ra': [-56, 10],
            'pm_dec': [11, 10],
            'v_rad': [189.6, 0.2],
            
            'fullname': 'Cen X-4'
        }
    }

    system = Sys_Dict[sysname]
    return system