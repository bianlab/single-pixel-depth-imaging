from __future__ import division

from utils.utils import *
from utils.datasets import *

import os
import sys
import time


import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

EPOCH = 100
BATCH_SIZE = 1
LR = 0.0001         # learning rate

f=open("./last_data.txt","r")
e=list(range(4500))
for number in range(4500):
   e[number]=list(range(10000))
   r=f.readline()
   print(number)
   e[number][0],e[number][1],e[number][2],e[number][3],e[number][4],e[number][5],e[number][6],e[number][7],e[number][8],e[number][9],\
   e[number][10],e[number][11],e[number][12],e[number][13],e[number][14],e[number][15],e[number][16],e[number][17],e[number][18],e[number][19],\
   e[number][20],e[number][21],e[number][22],e[number][23],e[number][24],e[number][25],e[number][26],e[number][27],e[number][28],e[number][29],\
   e[number][30],e[number][31],e[number][32],e[number][33],e[number][34],e[number][35],e[number][36],e[number][37],e[number][38],e[number][39],\
   e[number][40],e[number][41],e[number][42],e[number][43],e[number][44],e[number][45],e[number][46],e[number][47],e[number][48],e[number][49],\
   e[number][50],e[number][51],e[number][52],e[number][53],e[number][54],e[number][55],e[number][56],e[number][57],e[number][58],e[number][59],\
   e[number][60],e[number][61],e[number][62],e[number][63],e[number][64],e[number][65],e[number][66],e[number][67],e[number][68],e[number][69],\
   e[number][70],e[number][71],e[number][72],e[number][73],e[number][74],e[number][75],e[number][76],e[number][77],e[number][78],e[number][79],\
   e[number][80],e[number][81],e[number][82],e[number][83],e[number][84],e[number][85],e[number][86],e[number][87],e[number][88],e[number][89],\
   e[number][90],e[number][91],e[number][92],e[number][93],e[number][94],e[number][95],e[number][96],e[number][97],e[number][98],e[number][99],\
   e[number][100],e[number][101],e[number][102],e[number][103],e[number][104],e[number][105],e[number][106],e[number][107],e[number][108],e[number][109],\
   e[number][110],e[number][111],e[number][112],e[number][113],e[number][114],e[number][115],e[number][116],e[number][117],e[number][118],e[number][119],\
   e[number][120],e[number][121],e[number][122],e[number][123],e[number][124],e[number][125],e[number][126],e[number][127],e[number][128],e[number][129],\
   e[number][130],e[number][131],e[number][132],e[number][133],e[number][134],e[number][135],e[number][136],e[number][137],e[number][138],e[number][139],\
   e[number][140],e[number][141],e[number][142],e[number][143],e[number][144],e[number][145],e[number][146],e[number][147],e[number][148],e[number][149],\
   e[number][150],e[number][151],e[number][152],e[number][153],e[number][154],e[number][155],e[number][156],e[number][157],e[number][158],e[number][159],\
   e[number][160],e[number][161],e[number][162],e[number][163],e[number][164],e[number][165],e[number][166],e[number][167],e[number][168],e[number][169],\
   e[number][170],e[number][171],e[number][172],e[number][173],e[number][174],e[number][175],e[number][176],e[number][177],e[number][178],e[number][179],\
   e[number][180],e[number][181],e[number][182],e[number][183],e[number][184],e[number][185],e[number][186],e[number][187],e[number][188],e[number][189],\
   e[number][190],e[number][191],e[number][192],e[number][193],e[number][194],e[number][195],e[number][196],e[number][197],e[number][198],e[number][199],\
   e[number][200],e[number][201],e[number][202],e[number][203],e[number][204],e[number][205],e[number][206],e[number][207],e[number][208],e[number][209],\
   e[number][210],e[number][211],e[number][212],e[number][213],e[number][214],e[number][215],e[number][216],e[number][217],e[number][218],e[number][219],\
   e[number][220],e[number][221],e[number][222],e[number][223],e[number][224],e[number][225],e[number][226],e[number][227],e[number][228],e[number][229],\
   e[number][230],e[number][231],e[number][232],e[number][233],e[number][234],e[number][235],e[number][236],e[number][237],e[number][238],e[number][239],\
   e[number][240],e[number][241],e[number][242],e[number][243],e[number][244],e[number][245],e[number][246],e[number][247],e[number][248],e[number][249],\
   e[number][250],e[number][251],e[number][252],e[number][253],e[number][254],e[number][255],e[number][256],e[number][257],e[number][258],e[number][259],\
   e[number][260],e[number][261],e[number][262],e[number][263],e[number][264],e[number][265],e[number][266],e[number][267],e[number][268],e[number][269],\
   e[number][270],e[number][271],e[number][272],e[number][273],e[number][274],e[number][275],e[number][276],e[number][277],e[number][278],e[number][279],\
   e[number][280],e[number][281],e[number][282],e[number][283],e[number][284],e[number][285],e[number][286],e[number][287],e[number][288],e[number][289],\
   e[number][290],e[number][291],e[number][292],e[number][293],e[number][294],e[number][295],e[number][296],e[number][297],e[number][298],e[number][299],\
   e[number][300],e[number][301],e[number][302],e[number][303],e[number][304],e[number][305],e[number][306],e[number][307],e[number][308],e[number][309],\
   e[number][310],e[number][311],e[number][312],e[number][313],e[number][314],e[number][315],e[number][316],e[number][317],e[number][318],e[number][319],\
   e[number][320],e[number][321],e[number][322],e[number][323],e[number][324],e[number][325],e[number][326],e[number][327],e[number][328],e[number][329],\
   e[number][330],e[number][331],e[number][332],e[number][333],e[number][334],e[number][335],e[number][336],e[number][337],e[number][338],e[number][339],\
   e[number][340],e[number][341],e[number][342],e[number][343],e[number][344],e[number][345],e[number][346],e[number][347],e[number][348],e[number][349],\
   e[number][350],e[number][351],e[number][352],e[number][353],e[number][354],e[number][355],e[number][356],e[number][357],e[number][358],e[number][359],\
   e[number][360],e[number][361],e[number][362],e[number][363],e[number][364],e[number][365],e[number][366],e[number][367],e[number][368],e[number][369],\
   e[number][370],e[number][371],e[number][372],e[number][373],e[number][374],e[number][375],e[number][376],e[number][377],e[number][378],e[number][379],\
   e[number][380],e[number][381],e[number][382],e[number][383],e[number][384],e[number][385],e[number][386],e[number][387],e[number][388],e[number][389],\
   e[number][390],e[number][391],e[number][392],e[number][393],e[number][394],e[number][395],e[number][396],e[number][397],e[number][398],e[number][399],\
   e[number][400],e[number][401],e[number][402],e[number][403],e[number][404],e[number][405],e[number][406],e[number][407],e[number][408],e[number][409],\
   e[number][410],e[number][411],e[number][412],e[number][413],e[number][414],e[number][415],e[number][416],e[number][417],e[number][418],e[number][419],\
   e[number][420],e[number][421],e[number][422],e[number][423],e[number][424],e[number][425],e[number][426],e[number][427],e[number][428],e[number][429],\
   e[number][430],e[number][431],e[number][432],e[number][433],e[number][434],e[number][435],e[number][436],e[number][437],e[number][438],e[number][439],\
   e[number][440],e[number][441],e[number][442],e[number][443],e[number][444],e[number][445],e[number][446],e[number][447],e[number][448],e[number][449],\
   e[number][450],e[number][451],e[number][452],e[number][453],e[number][454],e[number][455],e[number][456],e[number][457],e[number][458],e[number][459],\
   e[number][460],e[number][461],e[number][462],e[number][463],e[number][464],e[number][465],e[number][466],e[number][467],e[number][468],e[number][469],\
   e[number][470],e[number][471],e[number][472],e[number][473],e[number][474],e[number][475],e[number][476],e[number][477],e[number][478],e[number][479],\
   e[number][480],e[number][481],e[number][482],e[number][483],e[number][484],e[number][485],e[number][486],e[number][487],e[number][488],e[number][489],\
   e[number][490],e[number][491],e[number][492],e[number][493],e[number][494],e[number][495],e[number][496],e[number][497],e[number][498],e[number][499],\
   e[number][500],e[number][501],e[number][502],e[number][503],e[number][504],e[number][505],e[number][506],e[number][507],e[number][508],e[number][509],\
   e[number][510],e[number][511],e[number][512],e[number][513],e[number][514],e[number][515],e[number][516],e[number][517],e[number][518],e[number][519],\
   e[number][520],e[number][521],e[number][522],e[number][523],e[number][524],e[number][525],e[number][526],e[number][527],e[number][528],e[number][529],\
   e[number][530],e[number][531],e[number][532],e[number][533],e[number][534],e[number][535],e[number][536],e[number][537],e[number][538],e[number][539],\
   e[number][540],e[number][541],e[number][542],e[number][543],e[number][544],e[number][545],e[number][546],e[number][547],e[number][548],e[number][549],\
   e[number][550],e[number][551],e[number][552],e[number][553],e[number][554],e[number][555],e[number][556],e[number][557],e[number][558],e[number][559],\
   e[number][560],e[number][561],e[number][562],e[number][563],e[number][564],e[number][565],e[number][566],e[number][567],e[number][568],e[number][569],\
   e[number][570],e[number][571],e[number][572],e[number][573],e[number][574],e[number][575],e[number][576],e[number][577],e[number][578],e[number][579],\
   e[number][580],e[number][581],e[number][582],e[number][583],e[number][584],e[number][585],e[number][586],e[number][587],e[number][588],e[number][589],\
   e[number][590],e[number][591],e[number][592],e[number][593],e[number][594],e[number][595],e[number][596],e[number][597],e[number][598],e[number][599],\
   e[number][600],e[number][601],e[number][602],e[number][603],e[number][604],e[number][605],e[number][606],e[number][607],e[number][608],e[number][609],\
   e[number][610],e[number][611],e[number][612],e[number][613],e[number][614],e[number][615],e[number][616],e[number][617],e[number][618],e[number][619],\
   e[number][620],e[number][621],e[number][622],e[number][623],e[number][624],e[number][625],e[number][626],e[number][627],e[number][628],e[number][629],\
   e[number][630],e[number][631],e[number][632],e[number][633],e[number][634],e[number][635],e[number][636],e[number][637],e[number][638],e[number][639],\
   e[number][640],e[number][641],e[number][642],e[number][643],e[number][644],e[number][645],e[number][646],e[number][647],e[number][648],e[number][649],\
   e[number][650],e[number][651],e[number][652],e[number][653],e[number][654],e[number][655],e[number][656],e[number][657],e[number][658],e[number][659],\
   e[number][660],e[number][661],e[number][662],e[number][663],e[number][664],e[number][665],e[number][666],e[number][667],e[number][668],e[number][669],\
   e[number][670],e[number][671],e[number][672],e[number][673],e[number][674],e[number][675],e[number][676],e[number][677],e[number][678],e[number][679],\
   e[number][680],e[number][681],e[number][682],e[number][683],e[number][684],e[number][685],e[number][686],e[number][687],e[number][688],e[number][689],\
   e[number][690],e[number][691],e[number][692],e[number][693],e[number][694],e[number][695],e[number][696],e[number][697],e[number][698],e[number][699],\
   e[number][700],e[number][701],e[number][702],e[number][703],e[number][704],e[number][705],e[number][706],e[number][707],e[number][708],e[number][709],\
   e[number][710],e[number][711],e[number][712],e[number][713],e[number][714],e[number][715],e[number][716],e[number][717],e[number][718],e[number][719],\
   e[number][720],e[number][721],e[number][722],e[number][723],e[number][724],e[number][725],e[number][726],e[number][727],e[number][728],e[number][729],\
   e[number][730],e[number][731],e[number][732],e[number][733],e[number][734],e[number][735],e[number][736],e[number][737],e[number][738],e[number][739],\
   e[number][740],e[number][741],e[number][742],e[number][743],e[number][744],e[number][745],e[number][746],e[number][747],e[number][748],e[number][749],\
   e[number][750],e[number][751],e[number][752],e[number][753],e[number][754],e[number][755],e[number][756],e[number][757],e[number][758],e[number][759],\
   e[number][760],e[number][761],e[number][762],e[number][763],e[number][764],e[number][765],e[number][766],e[number][767],e[number][768],e[number][769],\
   e[number][770],e[number][771],e[number][772],e[number][773],e[number][774],e[number][775],e[number][776],e[number][777],e[number][778],e[number][779],\
   e[number][780],e[number][781],e[number][782],e[number][783],e[number][784],e[number][785],e[number][786],e[number][787],e[number][788],e[number][789],\
   e[number][790],e[number][791],e[number][792],e[number][793],e[number][794],e[number][795],e[number][796],e[number][797],e[number][798],e[number][799],\
   e[number][800],e[number][801],e[number][802],e[number][803],e[number][804],e[number][805],e[number][806],e[number][807],e[number][808],e[number][809],\
   e[number][810],e[number][811],e[number][812],e[number][813],e[number][814],e[number][815],e[number][816],e[number][817],e[number][818],e[number][819],\
   e[number][820],e[number][821],e[number][822],e[number][823],e[number][824],e[number][825],e[number][826],e[number][827],e[number][828],e[number][829],\
   e[number][830],e[number][831],e[number][832],e[number][833],e[number][834],e[number][835],e[number][836],e[number][837],e[number][838],e[number][839],\
   e[number][840],e[number][841],e[number][842],e[number][843],e[number][844],e[number][845],e[number][846],e[number][847],e[number][848],e[number][849],\
   e[number][850],e[number][851],e[number][852],e[number][853],e[number][854],e[number][855],e[number][856],e[number][857],e[number][858],e[number][859],\
   e[number][860],e[number][861],e[number][862],e[number][863],e[number][864],e[number][865],e[number][866],e[number][867],e[number][868],e[number][869],\
   e[number][870],e[number][871],e[number][872],e[number][873],e[number][874],e[number][875],e[number][876],e[number][877],e[number][878],e[number][879],\
   e[number][880],e[number][881],e[number][882],e[number][883],e[number][884],e[number][885],e[number][886],e[number][887],e[number][888],e[number][889],\
   e[number][890],e[number][891],e[number][892],e[number][893],e[number][894],e[number][895],e[number][896],e[number][897],e[number][898],e[number][899],\
   e[number][900],e[number][901],e[number][902],e[number][903],e[number][904],e[number][905],e[number][906],e[number][907],e[number][908],e[number][909],\
   e[number][910],e[number][911],e[number][912],e[number][913],e[number][914],e[number][915],e[number][916],e[number][917],e[number][918],e[number][919],\
   e[number][920],e[number][921],e[number][922],e[number][923],e[number][924],e[number][925],e[number][926],e[number][927],e[number][928],e[number][929],\
   e[number][930],e[number][931],e[number][932],e[number][933],e[number][934],e[number][935],e[number][936],e[number][937],e[number][938],e[number][939],\
   e[number][940],e[number][941],e[number][942],e[number][943],e[number][944],e[number][945],e[number][946],e[number][947],e[number][948],e[number][949],\
   e[number][950],e[number][951],e[number][952],e[number][953],e[number][954],e[number][955],e[number][956],e[number][957],e[number][958],e[number][959],\
   e[number][960],e[number][961],e[number][962],e[number][963],e[number][964],e[number][965],e[number][966],e[number][967],e[number][968],e[number][969],\
   e[number][970],e[number][971],e[number][972],e[number][973],e[number][974],e[number][975],e[number][976],e[number][977],e[number][978],e[number][979],\
   e[number][980],e[number][981],e[number][982],e[number][983],e[number][984],e[number][985],e[number][986],e[number][987],e[number][988],e[number][989],\
   e[number][990],e[number][991],e[number][992],e[number][993],e[number][994],e[number][995],e[number][996],e[number][997],e[number][998],e[number][999],\
   e[number][1000],e[number][1001],e[number][1002],e[number][1003],e[number][1004],e[number][1005],e[number][1006],e[number][1007],e[number][1008],e[number][1009],\
   e[number][1010],e[number][1011],e[number][1012],e[number][1013],e[number][1014],e[number][1015],e[number][1016],e[number][1017],e[number][1018],e[number][1019],\
   e[number][1020],e[number][1021],e[number][1022],e[number][1023],e[number][1024],e[number][1025],e[number][1026],e[number][1027],e[number][1028],e[number][1029],\
   e[number][1030],e[number][1031],e[number][1032],e[number][1033],e[number][1034],e[number][1035],e[number][1036],e[number][1037],e[number][1038],e[number][1039],\
   e[number][1040],e[number][1041],e[number][1042],e[number][1043],e[number][1044],e[number][1045],e[number][1046],e[number][1047],e[number][1048],e[number][1049],\
   e[number][1050],e[number][1051],e[number][1052],e[number][1053],e[number][1054],e[number][1055],e[number][1056],e[number][1057],e[number][1058],e[number][1059],\
   e[number][1060],e[number][1061],e[number][1062],e[number][1063],e[number][1064],e[number][1065],e[number][1066],e[number][1067],e[number][1068],e[number][1069],\
   e[number][1070],e[number][1071],e[number][1072],e[number][1073],e[number][1074],e[number][1075],e[number][1076],e[number][1077],e[number][1078],e[number][1079],\
   e[number][1080],e[number][1081],e[number][1082],e[number][1083],e[number][1084],e[number][1085],e[number][1086],e[number][1087],e[number][1088],e[number][1089],\
   e[number][1090],e[number][1091],e[number][1092],e[number][1093],e[number][1094],e[number][1095],e[number][1096],e[number][1097],e[number][1098],e[number][1099],\
   e[number][1100],e[number][1101],e[number][1102],e[number][1103],e[number][1104],e[number][1105],e[number][1106],e[number][1107],e[number][1108],e[number][1109],\
   e[number][1110],e[number][1111],e[number][1112],e[number][1113],e[number][1114],e[number][1115],e[number][1116],e[number][1117],e[number][1118],e[number][1119],\
   e[number][1120],e[number][1121],e[number][1122],e[number][1123],e[number][1124],e[number][1125],e[number][1126],e[number][1127],e[number][1128],e[number][1129],\
   e[number][1130],e[number][1131],e[number][1132],e[number][1133],e[number][1134],e[number][1135],e[number][1136],e[number][1137],e[number][1138],e[number][1139],\
   e[number][1140],e[number][1141],e[number][1142],e[number][1143],e[number][1144],e[number][1145],e[number][1146],e[number][1147],e[number][1148],e[number][1149],\
   e[number][1150],e[number][1151],e[number][1152],e[number][1153],e[number][1154],e[number][1155],e[number][1156],e[number][1157],e[number][1158],e[number][1159],\
   e[number][1160],e[number][1161],e[number][1162],e[number][1163],e[number][1164],e[number][1165],e[number][1166],e[number][1167],e[number][1168],e[number][1169],\
   e[number][1170],e[number][1171],e[number][1172],e[number][1173],e[number][1174],e[number][1175],e[number][1176],e[number][1177],e[number][1178],e[number][1179],\
   e[number][1180],e[number][1181],e[number][1182],e[number][1183],e[number][1184],e[number][1185],e[number][1186],e[number][1187],e[number][1188],e[number][1189],\
   e[number][1190],e[number][1191],e[number][1192],e[number][1193],e[number][1194],e[number][1195],e[number][1196],e[number][1197],e[number][1198],e[number][1199],\
   e[number][1200],e[number][1201],e[number][1202],e[number][1203],e[number][1204],e[number][1205],e[number][1206],e[number][1207],e[number][1208],e[number][1209],\
   e[number][1210],e[number][1211],e[number][1212],e[number][1213],e[number][1214],e[number][1215],e[number][1216],e[number][1217],e[number][1218],e[number][1219],\
   e[number][1220],e[number][1221],e[number][1222],e[number][1223],e[number][1224],e[number][1225],e[number][1226],e[number][1227],e[number][1228],e[number][1229],\
   e[number][1230],e[number][1231],e[number][1232],e[number][1233],e[number][1234],e[number][1235],e[number][1236],e[number][1237],e[number][1238],e[number][1239],\
   e[number][1240],e[number][1241],e[number][1242],e[number][1243],e[number][1244],e[number][1245],e[number][1246],e[number][1247],e[number][1248],e[number][1249],\
   e[number][1250],e[number][1251],e[number][1252],e[number][1253],e[number][1254],e[number][1255],e[number][1256],e[number][1257],e[number][1258],e[number][1259],\
   e[number][1260],e[number][1261],e[number][1262],e[number][1263],e[number][1264],e[number][1265],e[number][1266],e[number][1267],e[number][1268],e[number][1269],\
   e[number][1270],e[number][1271],e[number][1272],e[number][1273],e[number][1274],e[number][1275],e[number][1276],e[number][1277],e[number][1278],e[number][1279],\
   e[number][1280],e[number][1281],e[number][1282],e[number][1283],e[number][1284],e[number][1285],e[number][1286],e[number][1287],e[number][1288],e[number][1289],\
   e[number][1290],e[number][1291],e[number][1292],e[number][1293],e[number][1294],e[number][1295],e[number][1296],e[number][1297],e[number][1298],e[number][1299],\
   e[number][1300],e[number][1301],e[number][1302],e[number][1303],e[number][1304],e[number][1305],e[number][1306],e[number][1307],e[number][1308],e[number][1309],\
   e[number][1310],e[number][1311],e[number][1312],e[number][1313],e[number][1314],e[number][1315],e[number][1316],e[number][1317],e[number][1318],e[number][1319],\
   e[number][1320],e[number][1321],e[number][1322],e[number][1323],e[number][1324],e[number][1325],e[number][1326],e[number][1327],e[number][1328],e[number][1329],\
   e[number][1330],e[number][1331],e[number][1332],e[number][1333],e[number][1334],e[number][1335],e[number][1336],e[number][1337],e[number][1338],e[number][1339],\
   e[number][1340],e[number][1341],e[number][1342],e[number][1343],e[number][1344],e[number][1345],e[number][1346],e[number][1347],e[number][1348],e[number][1349],\
   e[number][1350],e[number][1351],e[number][1352],e[number][1353],e[number][1354],e[number][1355],e[number][1356],e[number][1357],e[number][1358],e[number][1359],\
   e[number][1360],e[number][1361],e[number][1362],e[number][1363],e[number][1364],e[number][1365],e[number][1366],e[number][1367],e[number][1368],e[number][1369],\
   e[number][1370],e[number][1371],e[number][1372],e[number][1373],e[number][1374],e[number][1375],e[number][1376],e[number][1377],e[number][1378],e[number][1379],\
   e[number][1380],e[number][1381],e[number][1382],e[number][1383],e[number][1384],e[number][1385],e[number][1386],e[number][1387],e[number][1388],e[number][1389],\
   e[number][1390],e[number][1391],e[number][1392],e[number][1393],e[number][1394],e[number][1395],e[number][1396],e[number][1397],e[number][1398],e[number][1399],\
   e[number][1400],e[number][1401],e[number][1402],e[number][1403],e[number][1404],e[number][1405],e[number][1406],e[number][1407],e[number][1408],e[number][1409],\
   e[number][1410],e[number][1411],e[number][1412],e[number][1413],e[number][1414],e[number][1415],e[number][1416],e[number][1417],e[number][1418],e[number][1419],\
   e[number][1420],e[number][1421],e[number][1422],e[number][1423],e[number][1424],e[number][1425],e[number][1426],e[number][1427],e[number][1428],e[number][1429],\
   e[number][1430],e[number][1431],e[number][1432],e[number][1433],e[number][1434],e[number][1435],e[number][1436],e[number][1437],e[number][1438],e[number][1439],\
   e[number][1440],e[number][1441],e[number][1442],e[number][1443],e[number][1444],e[number][1445],e[number][1446],e[number][1447],e[number][1448],e[number][1449],\
   e[number][1450],e[number][1451],e[number][1452],e[number][1453],e[number][1454],e[number][1455],e[number][1456],e[number][1457],e[number][1458],e[number][1459],\
   e[number][1460],e[number][1461],e[number][1462],e[number][1463],e[number][1464],e[number][1465],e[number][1466],e[number][1467],e[number][1468],e[number][1469],\
   e[number][1470],e[number][1471],e[number][1472],e[number][1473],e[number][1474],e[number][1475],e[number][1476],e[number][1477],e[number][1478],e[number][1479],\
   e[number][1480],e[number][1481],e[number][1482],e[number][1483],e[number][1484],e[number][1485],e[number][1486],e[number][1487],e[number][1488],e[number][1489],\
   e[number][1490],e[number][1491],e[number][1492],e[number][1493],e[number][1494],e[number][1495],e[number][1496],e[number][1497],e[number][1498],e[number][1499],\
   e[number][1500],e[number][1501],e[number][1502],e[number][1503],e[number][1504],e[number][1505],e[number][1506],e[number][1507],e[number][1508],e[number][1509],\
   e[number][1510],e[number][1511],e[number][1512],e[number][1513],e[number][1514],e[number][1515],e[number][1516],e[number][1517],e[number][1518],e[number][1519],\
   e[number][1520],e[number][1521],e[number][1522],e[number][1523],e[number][1524],e[number][1525],e[number][1526],e[number][1527],e[number][1528],e[number][1529],\
   e[number][1530],e[number][1531],e[number][1532],e[number][1533],e[number][1534],e[number][1535],e[number][1536],e[number][1537],e[number][1538],e[number][1539],\
   e[number][1540],e[number][1541],e[number][1542],e[number][1543],e[number][1544],e[number][1545],e[number][1546],e[number][1547],e[number][1548],e[number][1549],\
   e[number][1550],e[number][1551],e[number][1552],e[number][1553],e[number][1554],e[number][1555],e[number][1556],e[number][1557],e[number][1558],e[number][1559],\
   e[number][1560],e[number][1561],e[number][1562],e[number][1563],e[number][1564],e[number][1565],e[number][1566],e[number][1567],e[number][1568],e[number][1569],\
   e[number][1570],e[number][1571],e[number][1572],e[number][1573],e[number][1574],e[number][1575],e[number][1576],e[number][1577],e[number][1578],e[number][1579],\
   e[number][1580],e[number][1581],e[number][1582],e[number][1583],e[number][1584],e[number][1585],e[number][1586],e[number][1587],e[number][1588],e[number][1589],\
   e[number][1590],e[number][1591],e[number][1592],e[number][1593],e[number][1594],e[number][1595],e[number][1596],e[number][1597],e[number][1598],e[number][1599],\
   e[number][1600],e[number][1601],e[number][1602],e[number][1603],e[number][1604],e[number][1605],e[number][1606],e[number][1607],e[number][1608],e[number][1609],\
   e[number][1610],e[number][1611],e[number][1612],e[number][1613],e[number][1614],e[number][1615],e[number][1616],e[number][1617],e[number][1618],e[number][1619],\
   e[number][1620],e[number][1621],e[number][1622],e[number][1623],e[number][1624],e[number][1625],e[number][1626],e[number][1627],e[number][1628],e[number][1629],\
   e[number][1630],e[number][1631],e[number][1632],e[number][1633],e[number][1634],e[number][1635],e[number][1636],e[number][1637],e[number][1638],e[number][1639],\
   e[number][1640],e[number][1641],e[number][1642],e[number][1643],e[number][1644],e[number][1645],e[number][1646],e[number][1647],e[number][1648],e[number][1649],\
   e[number][1650],e[number][1651],e[number][1652],e[number][1653],e[number][1654],e[number][1655],e[number][1656],e[number][1657],e[number][1658],e[number][1659],\
   e[number][1660],e[number][1661],e[number][1662],e[number][1663],e[number][1664],e[number][1665],e[number][1666],e[number][1667],e[number][1668],e[number][1669],\
   e[number][1670],e[number][1671],e[number][1672],e[number][1673],e[number][1674],e[number][1675],e[number][1676],e[number][1677],e[number][1678],e[number][1679],\
   e[number][1680],e[number][1681],e[number][1682],e[number][1683],e[number][1684],e[number][1685],e[number][1686],e[number][1687],e[number][1688],e[number][1689],\
   e[number][1690],e[number][1691],e[number][1692],e[number][1693],e[number][1694],e[number][1695],e[number][1696],e[number][1697],e[number][1698],e[number][1699],\
   e[number][1700],e[number][1701],e[number][1702],e[number][1703],e[number][1704],e[number][1705],e[number][1706],e[number][1707],e[number][1708],e[number][1709],\
   e[number][1710],e[number][1711],e[number][1712],e[number][1713],e[number][1714],e[number][1715],e[number][1716],e[number][1717],e[number][1718],e[number][1719],\
   e[number][1720],e[number][1721],e[number][1722],e[number][1723],e[number][1724],e[number][1725],e[number][1726],e[number][1727],e[number][1728],e[number][1729],\
   e[number][1730],e[number][1731],e[number][1732],e[number][1733],e[number][1734],e[number][1735],e[number][1736],e[number][1737],e[number][1738],e[number][1739],\
   e[number][1740],e[number][1741],e[number][1742],e[number][1743],e[number][1744],e[number][1745],e[number][1746],e[number][1747],e[number][1748],e[number][1749],\
   e[number][1750],e[number][1751],e[number][1752],e[number][1753],e[number][1754],e[number][1755],e[number][1756],e[number][1757],e[number][1758],e[number][1759],\
   e[number][1760],e[number][1761],e[number][1762],e[number][1763],e[number][1764],e[number][1765],e[number][1766],e[number][1767],e[number][1768],e[number][1769],\
   e[number][1770],e[number][1771],e[number][1772],e[number][1773],e[number][1774],e[number][1775],e[number][1776],e[number][1777],e[number][1778],e[number][1779],\
   e[number][1780],e[number][1781],e[number][1782],e[number][1783],e[number][1784],e[number][1785],e[number][1786],e[number][1787],e[number][1788],e[number][1789],\
   e[number][1790],e[number][1791],e[number][1792],e[number][1793],e[number][1794],e[number][1795],e[number][1796],e[number][1797],e[number][1798],e[number][1799],\
   e[number][1800],e[number][1801],e[number][1802],e[number][1803],e[number][1804],e[number][1805],e[number][1806],e[number][1807],e[number][1808],e[number][1809],\
   e[number][1810],e[number][1811],e[number][1812],e[number][1813],e[number][1814],e[number][1815],e[number][1816],e[number][1817],e[number][1818],e[number][1819],\
   e[number][1820],e[number][1821],e[number][1822],e[number][1823],e[number][1824],e[number][1825],e[number][1826],e[number][1827],e[number][1828],e[number][1829],\
   e[number][1830],e[number][1831],e[number][1832],e[number][1833],e[number][1834],e[number][1835],e[number][1836],e[number][1837],e[number][1838],e[number][1839],\
   e[number][1840],e[number][1841],e[number][1842],e[number][1843],e[number][1844],e[number][1845],e[number][1846],e[number][1847],e[number][1848],e[number][1849],\
   e[number][1850],e[number][1851],e[number][1852],e[number][1853],e[number][1854],e[number][1855],e[number][1856],e[number][1857],e[number][1858],e[number][1859],\
   e[number][1860],e[number][1861],e[number][1862],e[number][1863],e[number][1864],e[number][1865],e[number][1866],e[number][1867],e[number][1868],e[number][1869],\
   e[number][1870],e[number][1871],e[number][1872],e[number][1873],e[number][1874],e[number][1875],e[number][1876],e[number][1877],e[number][1878],e[number][1879],\
   e[number][1880],e[number][1881],e[number][1882],e[number][1883],e[number][1884],e[number][1885],e[number][1886],e[number][1887],e[number][1888],e[number][1889],\
   e[number][1890],e[number][1891],e[number][1892],e[number][1893],e[number][1894],e[number][1895],e[number][1896],e[number][1897],e[number][1898],e[number][1899],\
   e[number][1900],e[number][1901],e[number][1902],e[number][1903],e[number][1904],e[number][1905],e[number][1906],e[number][1907],e[number][1908],e[number][1909],\
   e[number][1910],e[number][1911],e[number][1912],e[number][1913],e[number][1914],e[number][1915],e[number][1916],e[number][1917],e[number][1918],e[number][1919],\
   e[number][1920],e[number][1921],e[number][1922],e[number][1923],e[number][1924],e[number][1925],e[number][1926],e[number][1927],e[number][1928],e[number][1929],\
   e[number][1930],e[number][1931],e[number][1932],e[number][1933],e[number][1934],e[number][1935],e[number][1936],e[number][1937],e[number][1938],e[number][1939],\
   e[number][1940],e[number][1941],e[number][1942],e[number][1943],e[number][1944],e[number][1945],e[number][1946],e[number][1947],e[number][1948],e[number][1949],\
   e[number][1950],e[number][1951],e[number][1952],e[number][1953],e[number][1954],e[number][1955],e[number][1956],e[number][1957],e[number][1958],e[number][1959],\
   e[number][1960],e[number][1961],e[number][1962],e[number][1963],e[number][1964],e[number][1965],e[number][1966],e[number][1967],e[number][1968],e[number][1969],\
   e[number][1970],e[number][1971],e[number][1972],e[number][1973],e[number][1974],e[number][1975],e[number][1976],e[number][1977],e[number][1978],e[number][1979],\
   e[number][1980],e[number][1981],e[number][1982],e[number][1983],e[number][1984],e[number][1985],e[number][1986],e[number][1987],e[number][1988],e[number][1989],\
   e[number][1990],e[number][1991],e[number][1992],e[number][1993],e[number][1994],e[number][1995],e[number][1996],e[number][1997],e[number][1998],e[number][1999],\
   e[number][2000],e[number][2001],e[number][2002],e[number][2003],e[number][2004],e[number][2005],e[number][2006],e[number][2007],e[number][2008],e[number][2009],\
   e[number][2010],e[number][2011],e[number][2012],e[number][2013],e[number][2014],e[number][2015],e[number][2016],e[number][2017],e[number][2018],e[number][2019],\
   e[number][2020],e[number][2021],e[number][2022],e[number][2023],e[number][2024],e[number][2025],e[number][2026],e[number][2027],e[number][2028],e[number][2029],\
   e[number][2030],e[number][2031],e[number][2032],e[number][2033],e[number][2034],e[number][2035],e[number][2036],e[number][2037],e[number][2038],e[number][2039],\
   e[number][2040],e[number][2041],e[number][2042],e[number][2043],e[number][2044],e[number][2045],e[number][2046],e[number][2047],e[number][2048],e[number][2049],\
   e[number][2050],e[number][2051],e[number][2052],e[number][2053],e[number][2054],e[number][2055],e[number][2056],e[number][2057],e[number][2058],e[number][2059],\
   e[number][2060],e[number][2061],e[number][2062],e[number][2063],e[number][2064],e[number][2065],e[number][2066],e[number][2067],e[number][2068],e[number][2069],\
   e[number][2070],e[number][2071],e[number][2072],e[number][2073],e[number][2074],e[number][2075],e[number][2076],e[number][2077],e[number][2078],e[number][2079],\
   e[number][2080],e[number][2081],e[number][2082],e[number][2083],e[number][2084],e[number][2085],e[number][2086],e[number][2087],e[number][2088],e[number][2089],\
   e[number][2090],e[number][2091],e[number][2092],e[number][2093],e[number][2094],e[number][2095],e[number][2096],e[number][2097],e[number][2098],e[number][2099],\
   e[number][2100],e[number][2101],e[number][2102],e[number][2103],e[number][2104],e[number][2105],e[number][2106],e[number][2107],e[number][2108],e[number][2109],\
   e[number][2110],e[number][2111],e[number][2112],e[number][2113],e[number][2114],e[number][2115],e[number][2116],e[number][2117],e[number][2118],e[number][2119],\
   e[number][2120],e[number][2121],e[number][2122],e[number][2123],e[number][2124],e[number][2125],e[number][2126],e[number][2127],e[number][2128],e[number][2129],\
   e[number][2130],e[number][2131],e[number][2132],e[number][2133],e[number][2134],e[number][2135],e[number][2136],e[number][2137],e[number][2138],e[number][2139],\
   e[number][2140],e[number][2141],e[number][2142],e[number][2143],e[number][2144],e[number][2145],e[number][2146],e[number][2147],e[number][2148],e[number][2149],\
   e[number][2150],e[number][2151],e[number][2152],e[number][2153],e[number][2154],e[number][2155],e[number][2156],e[number][2157],e[number][2158],e[number][2159],\
   e[number][2160],e[number][2161],e[number][2162],e[number][2163],e[number][2164],e[number][2165],e[number][2166],e[number][2167],e[number][2168],e[number][2169],\
   e[number][2170],e[number][2171],e[number][2172],e[number][2173],e[number][2174],e[number][2175],e[number][2176],e[number][2177],e[number][2178],e[number][2179],\
   e[number][2180],e[number][2181],e[number][2182],e[number][2183],e[number][2184],e[number][2185],e[number][2186],e[number][2187],e[number][2188],e[number][2189],\
   e[number][2190],e[number][2191],e[number][2192],e[number][2193],e[number][2194],e[number][2195],e[number][2196],e[number][2197],e[number][2198],e[number][2199],\
   e[number][2200],e[number][2201],e[number][2202],e[number][2203],e[number][2204],e[number][2205],e[number][2206],e[number][2207],e[number][2208],e[number][2209],\
   e[number][2210],e[number][2211],e[number][2212],e[number][2213],e[number][2214],e[number][2215],e[number][2216],e[number][2217],e[number][2218],e[number][2219],\
   e[number][2220],e[number][2221],e[number][2222],e[number][2223],e[number][2224],e[number][2225],e[number][2226],e[number][2227],e[number][2228],e[number][2229],\
   e[number][2230],e[number][2231],e[number][2232],e[number][2233],e[number][2234],e[number][2235],e[number][2236],e[number][2237],e[number][2238],e[number][2239],\
   e[number][2240],e[number][2241],e[number][2242],e[number][2243],e[number][2244],e[number][2245],e[number][2246],e[number][2247],e[number][2248],e[number][2249],\
   e[number][2250],e[number][2251],e[number][2252],e[number][2253],e[number][2254],e[number][2255],e[number][2256],e[number][2257],e[number][2258],e[number][2259],\
   e[number][2260],e[number][2261],e[number][2262],e[number][2263],e[number][2264],e[number][2265],e[number][2266],e[number][2267],e[number][2268],e[number][2269],\
   e[number][2270],e[number][2271],e[number][2272],e[number][2273],e[number][2274],e[number][2275],e[number][2276],e[number][2277],e[number][2278],e[number][2279],\
   e[number][2280],e[number][2281],e[number][2282],e[number][2283],e[number][2284],e[number][2285],e[number][2286],e[number][2287],e[number][2288],e[number][2289],\
   e[number][2290],e[number][2291],e[number][2292],e[number][2293],e[number][2294],e[number][2295],e[number][2296],e[number][2297],e[number][2298],e[number][2299],\
   e[number][2300],e[number][2301],e[number][2302],e[number][2303],e[number][2304],e[number][2305],e[number][2306],e[number][2307],e[number][2308],e[number][2309],\
   e[number][2310],e[number][2311],e[number][2312],e[number][2313],e[number][2314],e[number][2315],e[number][2316],e[number][2317],e[number][2318],e[number][2319],\
   e[number][2320],e[number][2321],e[number][2322],e[number][2323],e[number][2324],e[number][2325],e[number][2326],e[number][2327],e[number][2328],e[number][2329],\
   e[number][2330],e[number][2331],e[number][2332],e[number][2333],e[number][2334],e[number][2335],e[number][2336],e[number][2337],e[number][2338],e[number][2339],\
   e[number][2340],e[number][2341],e[number][2342],e[number][2343],e[number][2344],e[number][2345],e[number][2346],e[number][2347],e[number][2348],e[number][2349],\
   e[number][2350],e[number][2351],e[number][2352],e[number][2353],e[number][2354],e[number][2355],e[number][2356],e[number][2357],e[number][2358],e[number][2359],\
   e[number][2360],e[number][2361],e[number][2362],e[number][2363],e[number][2364],e[number][2365],e[number][2366],e[number][2367],e[number][2368],e[number][2369],\
   e[number][2370],e[number][2371],e[number][2372],e[number][2373],e[number][2374],e[number][2375],e[number][2376],e[number][2377],e[number][2378],e[number][2379],\
   e[number][2380],e[number][2381],e[number][2382],e[number][2383],e[number][2384],e[number][2385],e[number][2386],e[number][2387],e[number][2388],e[number][2389],\
   e[number][2390],e[number][2391],e[number][2392],e[number][2393],e[number][2394],e[number][2395],e[number][2396],e[number][2397],e[number][2398],e[number][2399],\
   e[number][2400],e[number][2401],e[number][2402],e[number][2403],e[number][2404],e[number][2405],e[number][2406],e[number][2407],e[number][2408],e[number][2409],\
   e[number][2410],e[number][2411],e[number][2412],e[number][2413],e[number][2414],e[number][2415],e[number][2416],e[number][2417],e[number][2418],e[number][2419],\
   e[number][2420],e[number][2421],e[number][2422],e[number][2423],e[number][2424],e[number][2425],e[number][2426],e[number][2427],e[number][2428],e[number][2429],\
   e[number][2430],e[number][2431],e[number][2432],e[number][2433],e[number][2434],e[number][2435],e[number][2436],e[number][2437],e[number][2438],e[number][2439],\
   e[number][2440],e[number][2441],e[number][2442],e[number][2443],e[number][2444],e[number][2445],e[number][2446],e[number][2447],e[number][2448],e[number][2449],\
   e[number][2450],e[number][2451],e[number][2452],e[number][2453],e[number][2454],e[number][2455],e[number][2456],e[number][2457],e[number][2458],e[number][2459],\
   e[number][2460],e[number][2461],e[number][2462],e[number][2463],e[number][2464],e[number][2465],e[number][2466],e[number][2467],e[number][2468],e[number][2469],\
   e[number][2470],e[number][2471],e[number][2472],e[number][2473],e[number][2474],e[number][2475],e[number][2476],e[number][2477],e[number][2478],e[number][2479],\
   e[number][2480],e[number][2481],e[number][2482],e[number][2483],e[number][2484],e[number][2485],e[number][2486],e[number][2487],e[number][2488],e[number][2489],\
   e[number][2490],e[number][2491],e[number][2492],e[number][2493],e[number][2494],e[number][2495],e[number][2496],e[number][2497],e[number][2498],e[number][2499],\
   e[number][2500],e[number][2501],e[number][2502],e[number][2503],e[number][2504],e[number][2505],e[number][2506],e[number][2507],e[number][2508],e[number][2509],\
   e[number][2510],e[number][2511],e[number][2512],e[number][2513],e[number][2514],e[number][2515],e[number][2516],e[number][2517],e[number][2518],e[number][2519],\
   e[number][2520],e[number][2521],e[number][2522],e[number][2523],e[number][2524],e[number][2525],e[number][2526],e[number][2527],e[number][2528],e[number][2529],\
   e[number][2530],e[number][2531],e[number][2532],e[number][2533],e[number][2534],e[number][2535],e[number][2536],e[number][2537],e[number][2538],e[number][2539],\
   e[number][2540],e[number][2541],e[number][2542],e[number][2543],e[number][2544],e[number][2545],e[number][2546],e[number][2547],e[number][2548],e[number][2549],\
   e[number][2550],e[number][2551],e[number][2552],e[number][2553],e[number][2554],e[number][2555],e[number][2556],e[number][2557],e[number][2558],e[number][2559],\
   e[number][2560],e[number][2561],e[number][2562],e[number][2563],e[number][2564],e[number][2565],e[number][2566],e[number][2567],e[number][2568],e[number][2569],\
   e[number][2570],e[number][2571],e[number][2572],e[number][2573],e[number][2574],e[number][2575],e[number][2576],e[number][2577],e[number][2578],e[number][2579],\
   e[number][2580],e[number][2581],e[number][2582],e[number][2583],e[number][2584],e[number][2585],e[number][2586],e[number][2587],e[number][2588],e[number][2589],\
   e[number][2590],e[number][2591],e[number][2592],e[number][2593],e[number][2594],e[number][2595],e[number][2596],e[number][2597],e[number][2598],e[number][2599],\
   e[number][2600],e[number][2601],e[number][2602],e[number][2603],e[number][2604],e[number][2605],e[number][2606],e[number][2607],e[number][2608],e[number][2609],\
   e[number][2610],e[number][2611],e[number][2612],e[number][2613],e[number][2614],e[number][2615],e[number][2616],e[number][2617],e[number][2618],e[number][2619],\
   e[number][2620],e[number][2621],e[number][2622],e[number][2623],e[number][2624],e[number][2625],e[number][2626],e[number][2627],e[number][2628],e[number][2629],\
   e[number][2630],e[number][2631],e[number][2632],e[number][2633],e[number][2634],e[number][2635],e[number][2636],e[number][2637],e[number][2638],e[number][2639],\
   e[number][2640],e[number][2641],e[number][2642],e[number][2643],e[number][2644],e[number][2645],e[number][2646],e[number][2647],e[number][2648],e[number][2649],\
   e[number][2650],e[number][2651],e[number][2652],e[number][2653],e[number][2654],e[number][2655],e[number][2656],e[number][2657],e[number][2658],e[number][2659],\
   e[number][2660],e[number][2661],e[number][2662],e[number][2663],e[number][2664],e[number][2665],e[number][2666],e[number][2667],e[number][2668],e[number][2669],\
   e[number][2670],e[number][2671],e[number][2672],e[number][2673],e[number][2674],e[number][2675],e[number][2676],e[number][2677],e[number][2678],e[number][2679],\
   e[number][2680],e[number][2681],e[number][2682],e[number][2683],e[number][2684],e[number][2685],e[number][2686],e[number][2687],e[number][2688],e[number][2689],\
   e[number][2690],e[number][2691],e[number][2692],e[number][2693],e[number][2694],e[number][2695],e[number][2696],e[number][2697],e[number][2698],e[number][2699],\
   e[number][2700],e[number][2701],e[number][2702],e[number][2703],e[number][2704],e[number][2705],e[number][2706],e[number][2707],e[number][2708],e[number][2709],\
   e[number][2710],e[number][2711],e[number][2712],e[number][2713],e[number][2714],e[number][2715],e[number][2716],e[number][2717],e[number][2718],e[number][2719],\
   e[number][2720],e[number][2721],e[number][2722],e[number][2723],e[number][2724],e[number][2725],e[number][2726],e[number][2727],e[number][2728],e[number][2729],\
   e[number][2730],e[number][2731],e[number][2732],e[number][2733],e[number][2734],e[number][2735],e[number][2736],e[number][2737],e[number][2738],e[number][2739],\
   e[number][2740],e[number][2741],e[number][2742],e[number][2743],e[number][2744],e[number][2745],e[number][2746],e[number][2747],e[number][2748],e[number][2749],\
   e[number][2750],e[number][2751],e[number][2752],e[number][2753],e[number][2754],e[number][2755],e[number][2756],e[number][2757],e[number][2758],e[number][2759],\
   e[number][2760],e[number][2761],e[number][2762],e[number][2763],e[number][2764],e[number][2765],e[number][2766],e[number][2767],e[number][2768],e[number][2769],\
   e[number][2770],e[number][2771],e[number][2772],e[number][2773],e[number][2774],e[number][2775],e[number][2776],e[number][2777],e[number][2778],e[number][2779],\
   e[number][2780],e[number][2781],e[number][2782],e[number][2783],e[number][2784],e[number][2785],e[number][2786],e[number][2787],e[number][2788],e[number][2789],\
   e[number][2790],e[number][2791],e[number][2792],e[number][2793],e[number][2794],e[number][2795],e[number][2796],e[number][2797],e[number][2798],e[number][2799],\
   e[number][2800],e[number][2801],e[number][2802],e[number][2803],e[number][2804],e[number][2805],e[number][2806],e[number][2807],e[number][2808],e[number][2809],\
   e[number][2810],e[number][2811],e[number][2812],e[number][2813],e[number][2814],e[number][2815],e[number][2816],e[number][2817],e[number][2818],e[number][2819],\
   e[number][2820],e[number][2821],e[number][2822],e[number][2823],e[number][2824],e[number][2825],e[number][2826],e[number][2827],e[number][2828],e[number][2829],\
   e[number][2830],e[number][2831],e[number][2832],e[number][2833],e[number][2834],e[number][2835],e[number][2836],e[number][2837],e[number][2838],e[number][2839],\
   e[number][2840],e[number][2841],e[number][2842],e[number][2843],e[number][2844],e[number][2845],e[number][2846],e[number][2847],e[number][2848],e[number][2849],\
   e[number][2850],e[number][2851],e[number][2852],e[number][2853],e[number][2854],e[number][2855],e[number][2856],e[number][2857],e[number][2858],e[number][2859],\
   e[number][2860],e[number][2861],e[number][2862],e[number][2863],e[number][2864],e[number][2865],e[number][2866],e[number][2867],e[number][2868],e[number][2869],\
   e[number][2870],e[number][2871],e[number][2872],e[number][2873],e[number][2874],e[number][2875],e[number][2876],e[number][2877],e[number][2878],e[number][2879],\
   e[number][2880],e[number][2881],e[number][2882],e[number][2883],e[number][2884],e[number][2885],e[number][2886],e[number][2887],e[number][2888],e[number][2889],\
   e[number][2890],e[number][2891],e[number][2892],e[number][2893],e[number][2894],e[number][2895],e[number][2896],e[number][2897],e[number][2898],e[number][2899],\
   e[number][2900],e[number][2901],e[number][2902],e[number][2903],e[number][2904],e[number][2905],e[number][2906],e[number][2907],e[number][2908],e[number][2909],\
   e[number][2910],e[number][2911],e[number][2912],e[number][2913],e[number][2914],e[number][2915],e[number][2916],e[number][2917],e[number][2918],e[number][2919],\
   e[number][2920],e[number][2921],e[number][2922],e[number][2923],e[number][2924],e[number][2925],e[number][2926],e[number][2927],e[number][2928],e[number][2929],\
   e[number][2930],e[number][2931],e[number][2932],e[number][2933],e[number][2934],e[number][2935],e[number][2936],e[number][2937],e[number][2938],e[number][2939],\
   e[number][2940],e[number][2941],e[number][2942],e[number][2943],e[number][2944],e[number][2945],e[number][2946],e[number][2947],e[number][2948],e[number][2949],\
   e[number][2950],e[number][2951],e[number][2952],e[number][2953],e[number][2954],e[number][2955],e[number][2956],e[number][2957],e[number][2958],e[number][2959],\
   e[number][2960],e[number][2961],e[number][2962],e[number][2963],e[number][2964],e[number][2965],e[number][2966],e[number][2967],e[number][2968],e[number][2969],\
   e[number][2970],e[number][2971],e[number][2972],e[number][2973],e[number][2974],e[number][2975],e[number][2976],e[number][2977],e[number][2978],e[number][2979],\
   e[number][2980],e[number][2981],e[number][2982],e[number][2983],e[number][2984],e[number][2985],e[number][2986],e[number][2987],e[number][2988],e[number][2989],\
   e[number][2990],e[number][2991],e[number][2992],e[number][2993],e[number][2994],e[number][2995],e[number][2996],e[number][2997],e[number][2998],e[number][2999],\
   e[number][3000],e[number][3001],e[number][3002],e[number][3003],e[number][3004],e[number][3005],e[number][3006],e[number][3007],e[number][3008],e[number][3009],\
   e[number][3010],e[number][3011],e[number][3012],e[number][3013],e[number][3014],e[number][3015],e[number][3016],e[number][3017],e[number][3018],e[number][3019],\
   e[number][3020],e[number][3021],e[number][3022],e[number][3023],e[number][3024],e[number][3025],e[number][3026],e[number][3027],e[number][3028],e[number][3029],\
   e[number][3030],e[number][3031],e[number][3032],e[number][3033],e[number][3034],e[number][3035],e[number][3036],e[number][3037],e[number][3038],e[number][3039],\
   e[number][3040],e[number][3041],e[number][3042],e[number][3043],e[number][3044],e[number][3045],e[number][3046],e[number][3047],e[number][3048],e[number][3049],\
   e[number][3050],e[number][3051],e[number][3052],e[number][3053],e[number][3054],e[number][3055],e[number][3056],e[number][3057],e[number][3058],e[number][3059],\
   e[number][3060],e[number][3061],e[number][3062],e[number][3063],e[number][3064],e[number][3065],e[number][3066],e[number][3067],e[number][3068],e[number][3069],\
   e[number][3070],e[number][3071],e[number][3072],e[number][3073],e[number][3074],e[number][3075],e[number][3076],e[number][3077],e[number][3078],e[number][3079],\
   e[number][3080],e[number][3081],e[number][3082],e[number][3083],e[number][3084],e[number][3085],e[number][3086],e[number][3087],e[number][3088],e[number][3089],\
   e[number][3090],e[number][3091],e[number][3092],e[number][3093],e[number][3094],e[number][3095],e[number][3096],e[number][3097],e[number][3098],e[number][3099],\
   e[number][3100],e[number][3101],e[number][3102],e[number][3103],e[number][3104],e[number][3105],e[number][3106],e[number][3107],e[number][3108],e[number][3109],\
   e[number][3110],e[number][3111],e[number][3112],e[number][3113],e[number][3114],e[number][3115],e[number][3116],e[number][3117],e[number][3118],e[number][3119],\
   e[number][3120],e[number][3121],e[number][3122],e[number][3123],e[number][3124],e[number][3125],e[number][3126],e[number][3127],e[number][3128],e[number][3129],\
   e[number][3130],e[number][3131],e[number][3132],e[number][3133],e[number][3134],e[number][3135],e[number][3136],e[number][3137],e[number][3138],e[number][3139],\
   e[number][3140],e[number][3141],e[number][3142],e[number][3143],e[number][3144],e[number][3145],e[number][3146],e[number][3147],e[number][3148],e[number][3149],\
   e[number][3150],e[number][3151],e[number][3152],e[number][3153],e[number][3154],e[number][3155],e[number][3156],e[number][3157],e[number][3158],e[number][3159],\
   e[number][3160],e[number][3161],e[number][3162],e[number][3163],e[number][3164],e[number][3165],e[number][3166],e[number][3167],e[number][3168],e[number][3169],\
   e[number][3170],e[number][3171],e[number][3172],e[number][3173],e[number][3174],e[number][3175],e[number][3176],e[number][3177],e[number][3178],e[number][3179],\
   e[number][3180],e[number][3181],e[number][3182],e[number][3183],e[number][3184],e[number][3185],e[number][3186],e[number][3187],e[number][3188],e[number][3189],\
   e[number][3190],e[number][3191],e[number][3192],e[number][3193],e[number][3194],e[number][3195],e[number][3196],e[number][3197],e[number][3198],e[number][3199],\
   e[number][3200],e[number][3201],e[number][3202],e[number][3203],e[number][3204],e[number][3205],e[number][3206],e[number][3207],e[number][3208],e[number][3209],\
   e[number][3210],e[number][3211],e[number][3212],e[number][3213],e[number][3214],e[number][3215],e[number][3216],e[number][3217],e[number][3218],e[number][3219],\
   e[number][3220],e[number][3221],e[number][3222],e[number][3223],e[number][3224],e[number][3225],e[number][3226],e[number][3227],e[number][3228],e[number][3229],\
   e[number][3230],e[number][3231],e[number][3232],e[number][3233],e[number][3234],e[number][3235],e[number][3236],e[number][3237],e[number][3238],e[number][3239],\
   e[number][3240],e[number][3241],e[number][3242],e[number][3243],e[number][3244],e[number][3245],e[number][3246],e[number][3247],e[number][3248],e[number][3249],\
   e[number][3250],e[number][3251],e[number][3252],e[number][3253],e[number][3254],e[number][3255],e[number][3256],e[number][3257],e[number][3258],e[number][3259],\
   e[number][3260],e[number][3261],e[number][3262],e[number][3263],e[number][3264],e[number][3265],e[number][3266],e[number][3267],e[number][3268],e[number][3269],\
   e[number][3270],e[number][3271],e[number][3272],e[number][3273],e[number][3274],e[number][3275],e[number][3276],e[number][3277],e[number][3278],e[number][3279],\
   e[number][3280],e[number][3281],e[number][3282],e[number][3283],e[number][3284],e[number][3285],e[number][3286],e[number][3287],e[number][3288],e[number][3289],\
   e[number][3290],e[number][3291],e[number][3292],e[number][3293],e[number][3294],e[number][3295],e[number][3296],e[number][3297],e[number][3298],e[number][3299],\
   e[number][3300],e[number][3301],e[number][3302],e[number][3303],e[number][3304],e[number][3305],e[number][3306],e[number][3307],e[number][3308],e[number][3309],\
   e[number][3310],e[number][3311],e[number][3312],e[number][3313],e[number][3314],e[number][3315],e[number][3316],e[number][3317],e[number][3318],e[number][3319],\
   e[number][3320],e[number][3321],e[number][3322],e[number][3323],e[number][3324],e[number][3325],e[number][3326],e[number][3327],e[number][3328],e[number][3329],\
   e[number][3330],e[number][3331],e[number][3332],e[number][3333],e[number][3334],e[number][3335],e[number][3336],e[number][3337],e[number][3338],e[number][3339],\
   e[number][3340],e[number][3341],e[number][3342],e[number][3343],e[number][3344],e[number][3345],e[number][3346],e[number][3347],e[number][3348],e[number][3349],\
   e[number][3350],e[number][3351],e[number][3352],e[number][3353],e[number][3354],e[number][3355],e[number][3356],e[number][3357],e[number][3358],e[number][3359],\
   e[number][3360],e[number][3361],e[number][3362],e[number][3363],e[number][3364],e[number][3365],e[number][3366],e[number][3367],e[number][3368],e[number][3369],\
   e[number][3370],e[number][3371],e[number][3372],e[number][3373],e[number][3374],e[number][3375],e[number][3376],e[number][3377],e[number][3378],e[number][3379],\
   e[number][3380],e[number][3381],e[number][3382],e[number][3383],e[number][3384],e[number][3385],e[number][3386],e[number][3387],e[number][3388],e[number][3389],\
   e[number][3390],e[number][3391],e[number][3392],e[number][3393],e[number][3394],e[number][3395],e[number][3396],e[number][3397],e[number][3398],e[number][3399],\
   e[number][3400],e[number][3401],e[number][3402],e[number][3403],e[number][3404],e[number][3405],e[number][3406],e[number][3407],e[number][3408],e[number][3409],\
   e[number][3410],e[number][3411],e[number][3412],e[number][3413],e[number][3414],e[number][3415],e[number][3416],e[number][3417],e[number][3418],e[number][3419],\
   e[number][3420],e[number][3421],e[number][3422],e[number][3423],e[number][3424],e[number][3425],e[number][3426],e[number][3427],e[number][3428],e[number][3429],\
   e[number][3430],e[number][3431],e[number][3432],e[number][3433],e[number][3434],e[number][3435],e[number][3436],e[number][3437],e[number][3438],e[number][3439],\
   e[number][3440],e[number][3441],e[number][3442],e[number][3443],e[number][3444],e[number][3445],e[number][3446],e[number][3447],e[number][3448],e[number][3449],\
   e[number][3450],e[number][3451],e[number][3452],e[number][3453],e[number][3454],e[number][3455],e[number][3456],e[number][3457],e[number][3458],e[number][3459],\
   e[number][3460],e[number][3461],e[number][3462],e[number][3463],e[number][3464],e[number][3465],e[number][3466],e[number][3467],e[number][3468],e[number][3469],\
   e[number][3470],e[number][3471],e[number][3472],e[number][3473],e[number][3474],e[number][3475],e[number][3476],e[number][3477],e[number][3478],e[number][3479],\
   e[number][3480],e[number][3481],e[number][3482],e[number][3483],e[number][3484],e[number][3485],e[number][3486],e[number][3487],e[number][3488],e[number][3489],\
   e[number][3490],e[number][3491],e[number][3492],e[number][3493],e[number][3494],e[number][3495],e[number][3496],e[number][3497],e[number][3498],e[number][3499],\
   e[number][3500],e[number][3501],e[number][3502],e[number][3503],e[number][3504],e[number][3505],e[number][3506],e[number][3507],e[number][3508],e[number][3509],\
   e[number][3510],e[number][3511],e[number][3512],e[number][3513],e[number][3514],e[number][3515],e[number][3516],e[number][3517],e[number][3518],e[number][3519],\
   e[number][3520],e[number][3521],e[number][3522],e[number][3523],e[number][3524],e[number][3525],e[number][3526],e[number][3527],e[number][3528],e[number][3529],\
   e[number][3530],e[number][3531],e[number][3532],e[number][3533],e[number][3534],e[number][3535],e[number][3536],e[number][3537],e[number][3538],e[number][3539],\
   e[number][3540],e[number][3541],e[number][3542],e[number][3543],e[number][3544],e[number][3545],e[number][3546],e[number][3547],e[number][3548],e[number][3549],\
   e[number][3550],e[number][3551],e[number][3552],e[number][3553],e[number][3554],e[number][3555],e[number][3556],e[number][3557],e[number][3558],e[number][3559],\
   e[number][3560],e[number][3561],e[number][3562],e[number][3563],e[number][3564],e[number][3565],e[number][3566],e[number][3567],e[number][3568],e[number][3569],\
   e[number][3570],e[number][3571],e[number][3572],e[number][3573],e[number][3574],e[number][3575],e[number][3576],e[number][3577],e[number][3578],e[number][3579],\
   e[number][3580],e[number][3581],e[number][3582],e[number][3583],e[number][3584],e[number][3585],e[number][3586],e[number][3587],e[number][3588],e[number][3589],\
   e[number][3590],e[number][3591],e[number][3592],e[number][3593],e[number][3594],e[number][3595],e[number][3596],e[number][3597],e[number][3598],e[number][3599],\
   e[number][3600],e[number][3601],e[number][3602],e[number][3603],e[number][3604],e[number][3605],e[number][3606],e[number][3607],e[number][3608],e[number][3609],\
   e[number][3610],e[number][3611],e[number][3612],e[number][3613],e[number][3614],e[number][3615],e[number][3616],e[number][3617],e[number][3618],e[number][3619],\
   e[number][3620],e[number][3621],e[number][3622],e[number][3623],e[number][3624],e[number][3625],e[number][3626],e[number][3627],e[number][3628],e[number][3629],\
   e[number][3630],e[number][3631],e[number][3632],e[number][3633],e[number][3634],e[number][3635],e[number][3636],e[number][3637],e[number][3638],e[number][3639],\
   e[number][3640],e[number][3641],e[number][3642],e[number][3643],e[number][3644],e[number][3645],e[number][3646],e[number][3647],e[number][3648],e[number][3649],\
   e[number][3650],e[number][3651],e[number][3652],e[number][3653],e[number][3654],e[number][3655],e[number][3656],e[number][3657],e[number][3658],e[number][3659],\
   e[number][3660],e[number][3661],e[number][3662],e[number][3663],e[number][3664],e[number][3665],e[number][3666],e[number][3667],e[number][3668],e[number][3669],\
   e[number][3670],e[number][3671],e[number][3672],e[number][3673],e[number][3674],e[number][3675],e[number][3676],e[number][3677],e[number][3678],e[number][3679],\
   e[number][3680],e[number][3681],e[number][3682],e[number][3683],e[number][3684],e[number][3685],e[number][3686],e[number][3687],e[number][3688],e[number][3689],\
   e[number][3690],e[number][3691],e[number][3692],e[number][3693],e[number][3694],e[number][3695],e[number][3696],e[number][3697],e[number][3698],e[number][3699],\
   e[number][3700],e[number][3701],e[number][3702],e[number][3703],e[number][3704],e[number][3705],e[number][3706],e[number][3707],e[number][3708],e[number][3709],\
   e[number][3710],e[number][3711],e[number][3712],e[number][3713],e[number][3714],e[number][3715],e[number][3716],e[number][3717],e[number][3718],e[number][3719],\
   e[number][3720],e[number][3721],e[number][3722],e[number][3723],e[number][3724],e[number][3725],e[number][3726],e[number][3727],e[number][3728],e[number][3729],\
   e[number][3730],e[number][3731],e[number][3732],e[number][3733],e[number][3734],e[number][3735],e[number][3736],e[number][3737],e[number][3738],e[number][3739],\
   e[number][3740],e[number][3741],e[number][3742],e[number][3743],e[number][3744],e[number][3745],e[number][3746],e[number][3747],e[number][3748],e[number][3749],\
   e[number][3750],e[number][3751],e[number][3752],e[number][3753],e[number][3754],e[number][3755],e[number][3756],e[number][3757],e[number][3758],e[number][3759],\
   e[number][3760],e[number][3761],e[number][3762],e[number][3763],e[number][3764],e[number][3765],e[number][3766],e[number][3767],e[number][3768],e[number][3769],\
   e[number][3770],e[number][3771],e[number][3772],e[number][3773],e[number][3774],e[number][3775],e[number][3776],e[number][3777],e[number][3778],e[number][3779],\
   e[number][3780],e[number][3781],e[number][3782],e[number][3783],e[number][3784],e[number][3785],e[number][3786],e[number][3787],e[number][3788],e[number][3789],\
   e[number][3790],e[number][3791],e[number][3792],e[number][3793],e[number][3794],e[number][3795],e[number][3796],e[number][3797],e[number][3798],e[number][3799],\
   e[number][3800],e[number][3801],e[number][3802],e[number][3803],e[number][3804],e[number][3805],e[number][3806],e[number][3807],e[number][3808],e[number][3809],\
   e[number][3810],e[number][3811],e[number][3812],e[number][3813],e[number][3814],e[number][3815],e[number][3816],e[number][3817],e[number][3818],e[number][3819],\
   e[number][3820],e[number][3821],e[number][3822],e[number][3823],e[number][3824],e[number][3825],e[number][3826],e[number][3827],e[number][3828],e[number][3829],\
   e[number][3830],e[number][3831],e[number][3832],e[number][3833],e[number][3834],e[number][3835],e[number][3836],e[number][3837],e[number][3838],e[number][3839],\
   e[number][3840],e[number][3841],e[number][3842],e[number][3843],e[number][3844],e[number][3845],e[number][3846],e[number][3847],e[number][3848],e[number][3849],\
   e[number][3850],e[number][3851],e[number][3852],e[number][3853],e[number][3854],e[number][3855],e[number][3856],e[number][3857],e[number][3858],e[number][3859],\
   e[number][3860],e[number][3861],e[number][3862],e[number][3863],e[number][3864],e[number][3865],e[number][3866],e[number][3867],e[number][3868],e[number][3869],\
   e[number][3870],e[number][3871],e[number][3872],e[number][3873],e[number][3874],e[number][3875],e[number][3876],e[number][3877],e[number][3878],e[number][3879],\
   e[number][3880],e[number][3881],e[number][3882],e[number][3883],e[number][3884],e[number][3885],e[number][3886],e[number][3887],e[number][3888],e[number][3889],\
   e[number][3890],e[number][3891],e[number][3892],e[number][3893],e[number][3894],e[number][3895],e[number][3896],e[number][3897],e[number][3898],e[number][3899],\
   e[number][3900],e[number][3901],e[number][3902],e[number][3903],e[number][3904],e[number][3905],e[number][3906],e[number][3907],e[number][3908],e[number][3909],\
   e[number][3910],e[number][3911],e[number][3912],e[number][3913],e[number][3914],e[number][3915],e[number][3916],e[number][3917],e[number][3918],e[number][3919],\
   e[number][3920],e[number][3921],e[number][3922],e[number][3923],e[number][3924],e[number][3925],e[number][3926],e[number][3927],e[number][3928],e[number][3929],\
   e[number][3930],e[number][3931],e[number][3932],e[number][3933],e[number][3934],e[number][3935],e[number][3936],e[number][3937],e[number][3938],e[number][3939],\
   e[number][3940],e[number][3941],e[number][3942],e[number][3943],e[number][3944],e[number][3945],e[number][3946],e[number][3947],e[number][3948],e[number][3949],\
   e[number][3950],e[number][3951],e[number][3952],e[number][3953],e[number][3954],e[number][3955],e[number][3956],e[number][3957],e[number][3958],e[number][3959],\
   e[number][3960],e[number][3961],e[number][3962],e[number][3963],e[number][3964],e[number][3965],e[number][3966],e[number][3967],e[number][3968],e[number][3969],\
   e[number][3970],e[number][3971],e[number][3972],e[number][3973],e[number][3974],e[number][3975],e[number][3976],e[number][3977],e[number][3978],e[number][3979],\
   e[number][3980],e[number][3981],e[number][3982],e[number][3983],e[number][3984],e[number][3985],e[number][3986],e[number][3987],e[number][3988],e[number][3989],\
   e[number][3990],e[number][3991],e[number][3992],e[number][3993],e[number][3994],e[number][3995],e[number][3996],e[number][3997],e[number][3998],e[number][3999],e[number][4000],\
e[number][4001],e[number][4002],e[number][4003],e[number][4004],e[number][4005],e[number][4006],e[number][4007],e[number][4008],e[number][4009],e[number][4010],\
e[number][4011],e[number][4012],e[number][4013],e[number][4014],e[number][4015],e[number][4016],e[number][4017],e[number][4018],e[number][4019],e[number][4020],\
e[number][4021],e[number][4022],e[number][4023],e[number][4024],e[number][4025],e[number][4026],e[number][4027],e[number][4028],e[number][4029],e[number][4030],\
e[number][4031],e[number][4032],e[number][4033],e[number][4034],e[number][4035],e[number][4036],e[number][4037],e[number][4038],e[number][4039],e[number][4040],\
e[number][4041],e[number][4042],e[number][4043],e[number][4044],e[number][4045],e[number][4046],e[number][4047],e[number][4048],e[number][4049],e[number][4050],\
e[number][4051],e[number][4052],e[number][4053],e[number][4054],e[number][4055],e[number][4056],e[number][4057],e[number][4058],e[number][4059],e[number][4060],\
e[number][4061],e[number][4062],e[number][4063],e[number][4064],e[number][4065],e[number][4066],e[number][4067],e[number][4068],e[number][4069],e[number][4070],\
e[number][4071],e[number][4072],e[number][4073],e[number][4074],e[number][4075],e[number][4076],e[number][4077],e[number][4078],e[number][4079],e[number][4080],\
e[number][4081],e[number][4082],e[number][4083],e[number][4084],e[number][4085],e[number][4086],e[number][4087],e[number][4088],e[number][4089],e[number][4090],\
e[number][4091],e[number][4092],e[number][4093],e[number][4094],e[number][4095],e[number][4096],e[number][4097],e[number][4098],e[number][4099],e[number][4100],\
e[number][4101],e[number][4102],e[number][4103],e[number][4104],e[number][4105],e[number][4106],e[number][4107],e[number][4108],e[number][4109],e[number][4110],\
e[number][4111],e[number][4112],e[number][4113],e[number][4114],e[number][4115],e[number][4116],e[number][4117],e[number][4118],e[number][4119],e[number][4120],\
e[number][4121],e[number][4122],e[number][4123],e[number][4124],e[number][4125],e[number][4126],e[number][4127],e[number][4128],e[number][4129],e[number][4130],\
e[number][4131],e[number][4132],e[number][4133],e[number][4134],e[number][4135],e[number][4136],e[number][4137],e[number][4138],e[number][4139],e[number][4140],\
e[number][4141],e[number][4142],e[number][4143],e[number][4144],e[number][4145],e[number][4146],e[number][4147],e[number][4148],e[number][4149],e[number][4150],\
e[number][4151],e[number][4152],e[number][4153],e[number][4154],e[number][4155],e[number][4156],e[number][4157],e[number][4158],e[number][4159],e[number][4160],\
e[number][4161],e[number][4162],e[number][4163],e[number][4164],e[number][4165],e[number][4166],e[number][4167],e[number][4168],e[number][4169],e[number][4170],\
e[number][4171],e[number][4172],e[number][4173],e[number][4174],e[number][4175],e[number][4176],e[number][4177],e[number][4178],e[number][4179],e[number][4180],\
e[number][4181],e[number][4182],e[number][4183],e[number][4184],e[number][4185],e[number][4186],e[number][4187],e[number][4188],e[number][4189],e[number][4190],\
e[number][4191],e[number][4192],e[number][4193],e[number][4194],e[number][4195],e[number][4196],e[number][4197],e[number][4198],e[number][4199],e[number][4200],\
e[number][4201],e[number][4202],e[number][4203],e[number][4204],e[number][4205],e[number][4206],e[number][4207],e[number][4208],e[number][4209],e[number][4210],\
e[number][4211],e[number][4212],e[number][4213],e[number][4214],e[number][4215],e[number][4216],e[number][4217],e[number][4218],e[number][4219],e[number][4220],\
e[number][4221],e[number][4222],e[number][4223],e[number][4224],e[number][4225],e[number][4226],e[number][4227],e[number][4228],e[number][4229],e[number][4230],\
e[number][4231],e[number][4232],e[number][4233],e[number][4234],e[number][4235],e[number][4236],e[number][4237],e[number][4238],e[number][4239],e[number][4240],\
e[number][4241],e[number][4242],e[number][4243],e[number][4244],e[number][4245],e[number][4246],e[number][4247],e[number][4248],e[number][4249],e[number][4250],\
e[number][4251],e[number][4252],e[number][4253],e[number][4254],e[number][4255],e[number][4256],e[number][4257],e[number][4258],e[number][4259],e[number][4260],\
e[number][4261],e[number][4262],e[number][4263],e[number][4264],e[number][4265],e[number][4266],e[number][4267],e[number][4268],e[number][4269],e[number][4270],\
e[number][4271],e[number][4272],e[number][4273],e[number][4274],e[number][4275],e[number][4276],e[number][4277],e[number][4278],e[number][4279],e[number][4280],\
e[number][4281],e[number][4282],e[number][4283],e[number][4284],e[number][4285],e[number][4286],e[number][4287],e[number][4288],e[number][4289],e[number][4290],\
e[number][4291],e[number][4292],e[number][4293],e[number][4294],e[number][4295],e[number][4296],e[number][4297],e[number][4298],e[number][4299],e[number][4300],\
e[number][4301],e[number][4302],e[number][4303],e[number][4304],e[number][4305],e[number][4306],e[number][4307],e[number][4308],e[number][4309],e[number][4310],\
e[number][4311],e[number][4312],e[number][4313],e[number][4314],e[number][4315],e[number][4316],e[number][4317],e[number][4318],e[number][4319],e[number][4320],\
e[number][4321],e[number][4322],e[number][4323],e[number][4324],e[number][4325],e[number][4326],e[number][4327],e[number][4328],e[number][4329],e[number][4330],\
e[number][4331],e[number][4332],e[number][4333],e[number][4334],e[number][4335],e[number][4336],e[number][4337],e[number][4338],e[number][4339],e[number][4340],\
e[number][4341],e[number][4342],e[number][4343],e[number][4344],e[number][4345],e[number][4346],e[number][4347],e[number][4348],e[number][4349],e[number][4350],\
e[number][4351],e[number][4352],e[number][4353],e[number][4354],e[number][4355],e[number][4356],e[number][4357],e[number][4358],e[number][4359],e[number][4360],\
e[number][4361],e[number][4362],e[number][4363],e[number][4364],e[number][4365],e[number][4366],e[number][4367],e[number][4368],e[number][4369],e[number][4370],\
e[number][4371],e[number][4372],e[number][4373],e[number][4374],e[number][4375],e[number][4376],e[number][4377],e[number][4378],e[number][4379],e[number][4380],\
e[number][4381],e[number][4382],e[number][4383],e[number][4384],e[number][4385],e[number][4386],e[number][4387],e[number][4388],e[number][4389],e[number][4390],\
e[number][4391],e[number][4392],e[number][4393],e[number][4394],e[number][4395],e[number][4396],e[number][4397],e[number][4398],e[number][4399],e[number][4400],\
e[number][4401],e[number][4402],e[number][4403],e[number][4404],e[number][4405],e[number][4406],e[number][4407],e[number][4408],e[number][4409],e[number][4410],\
e[number][4411],e[number][4412],e[number][4413],e[number][4414],e[number][4415],e[number][4416],e[number][4417],e[number][4418],e[number][4419],e[number][4420],\
e[number][4421],e[number][4422],e[number][4423],e[number][4424],e[number][4425],e[number][4426],e[number][4427],e[number][4428],e[number][4429],e[number][4430],\
e[number][4431],e[number][4432],e[number][4433],e[number][4434],e[number][4435],e[number][4436],e[number][4437],e[number][4438],e[number][4439],e[number][4440],\
e[number][4441],e[number][4442],e[number][4443],e[number][4444],e[number][4445],e[number][4446],e[number][4447],e[number][4448],e[number][4449],e[number][4450],\
e[number][4451],e[number][4452],e[number][4453],e[number][4454],e[number][4455],e[number][4456],e[number][4457],e[number][4458],e[number][4459],e[number][4460],\
e[number][4461],e[number][4462],e[number][4463],e[number][4464],e[number][4465],e[number][4466],e[number][4467],e[number][4468],e[number][4469],e[number][4470],\
e[number][4471],e[number][4472],e[number][4473],e[number][4474],e[number][4475],e[number][4476],e[number][4477],e[number][4478],e[number][4479],e[number][4480],\
e[number][4481],e[number][4482],e[number][4483],e[number][4484],e[number][4485],e[number][4486],e[number][4487],e[number][4488],e[number][4489],e[number][4490],\
e[number][4491],e[number][4492],e[number][4493],e[number][4494],e[number][4495],e[number][4496],e[number][4497],e[number][4498],e[number][4499],e[number][4500],\
e[number][4501],e[number][4502],e[number][4503],e[number][4504],e[number][4505],e[number][4506],e[number][4507],e[number][4508],e[number][4509],e[number][4510],\
e[number][4511],e[number][4512],e[number][4513],e[number][4514],e[number][4515],e[number][4516],e[number][4517],e[number][4518],e[number][4519],e[number][4520],\
e[number][4521],e[number][4522],e[number][4523],e[number][4524],e[number][4525],e[number][4526],e[number][4527],e[number][4528],e[number][4529],e[number][4530],\
e[number][4531],e[number][4532],e[number][4533],e[number][4534],e[number][4535],e[number][4536],e[number][4537],e[number][4538],e[number][4539],e[number][4540],\
e[number][4541],e[number][4542],e[number][4543],e[number][4544],e[number][4545],e[number][4546],e[number][4547],e[number][4548],e[number][4549],e[number][4550],\
e[number][4551],e[number][4552],e[number][4553],e[number][4554],e[number][4555],e[number][4556],e[number][4557],e[number][4558],e[number][4559],e[number][4560],\
e[number][4561],e[number][4562],e[number][4563],e[number][4564],e[number][4565],e[number][4566],e[number][4567],e[number][4568],e[number][4569],e[number][4570],\
e[number][4571],e[number][4572],e[number][4573],e[number][4574],e[number][4575],e[number][4576],e[number][4577],e[number][4578],e[number][4579],e[number][4580],\
e[number][4581],e[number][4582],e[number][4583],e[number][4584],e[number][4585],e[number][4586],e[number][4587],e[number][4588],e[number][4589],e[number][4590],\
e[number][4591],e[number][4592],e[number][4593],e[number][4594],e[number][4595],e[number][4596],e[number][4597],e[number][4598],e[number][4599],e[number][4600],\
e[number][4601],e[number][4602],e[number][4603],e[number][4604],e[number][4605],e[number][4606],e[number][4607],e[number][4608],e[number][4609],e[number][4610],\
e[number][4611],e[number][4612],e[number][4613],e[number][4614],e[number][4615],e[number][4616],e[number][4617],e[number][4618],e[number][4619],e[number][4620],\
e[number][4621],e[number][4622],e[number][4623],e[number][4624],e[number][4625],e[number][4626],e[number][4627],e[number][4628],e[number][4629],e[number][4630],\
e[number][4631],e[number][4632],e[number][4633],e[number][4634],e[number][4635],e[number][4636],e[number][4637],e[number][4638],e[number][4639],e[number][4640],\
e[number][4641],e[number][4642],e[number][4643],e[number][4644],e[number][4645],e[number][4646],e[number][4647],e[number][4648],e[number][4649],e[number][4650],\
e[number][4651],e[number][4652],e[number][4653],e[number][4654],e[number][4655],e[number][4656],e[number][4657],e[number][4658],e[number][4659],e[number][4660],\
e[number][4661],e[number][4662],e[number][4663],e[number][4664],e[number][4665],e[number][4666],e[number][4667],e[number][4668],e[number][4669],e[number][4670],\
e[number][4671],e[number][4672],e[number][4673],e[number][4674],e[number][4675],e[number][4676],e[number][4677],e[number][4678],e[number][4679],e[number][4680],\
e[number][4681],e[number][4682],e[number][4683],e[number][4684],e[number][4685],e[number][4686],e[number][4687],e[number][4688],e[number][4689],e[number][4690],\
e[number][4691],e[number][4692],e[number][4693],e[number][4694],e[number][4695],e[number][4696],e[number][4697],e[number][4698],e[number][4699],e[number][4700],\
e[number][4701],e[number][4702],e[number][4703],e[number][4704],e[number][4705],e[number][4706],e[number][4707],e[number][4708],e[number][4709],e[number][4710],\
e[number][4711],e[number][4712],e[number][4713],e[number][4714],e[number][4715],e[number][4716],e[number][4717],e[number][4718],e[number][4719],e[number][4720],\
e[number][4721],e[number][4722],e[number][4723],e[number][4724],e[number][4725],e[number][4726],e[number][4727],e[number][4728],e[number][4729],e[number][4730],\
e[number][4731],e[number][4732],e[number][4733],e[number][4734],e[number][4735],e[number][4736],e[number][4737],e[number][4738],e[number][4739],e[number][4740],\
e[number][4741],e[number][4742],e[number][4743],e[number][4744],e[number][4745],e[number][4746],e[number][4747],e[number][4748],e[number][4749],e[number][4750],\
e[number][4751],e[number][4752],e[number][4753],e[number][4754],e[number][4755],e[number][4756],e[number][4757],e[number][4758],e[number][4759],e[number][4760],\
e[number][4761],e[number][4762],e[number][4763],e[number][4764],e[number][4765],e[number][4766],e[number][4767],e[number][4768],e[number][4769],e[number][4770],\
e[number][4771],e[number][4772],e[number][4773],e[number][4774],e[number][4775],e[number][4776],e[number][4777],e[number][4778],e[number][4779],e[number][4780],\
e[number][4781],e[number][4782],e[number][4783],e[number][4784],e[number][4785],e[number][4786],e[number][4787],e[number][4788],e[number][4789],e[number][4790],\
e[number][4791],e[number][4792],e[number][4793],e[number][4794],e[number][4795],e[number][4796],e[number][4797],e[number][4798],e[number][4799],e[number][4800],\
e[number][4801],e[number][4802],e[number][4803],e[number][4804],e[number][4805],e[number][4806],e[number][4807],e[number][4808],e[number][4809],e[number][4810],\
e[number][4811],e[number][4812],e[number][4813],e[number][4814],e[number][4815],e[number][4816],e[number][4817],e[number][4818],e[number][4819],e[number][4820],\
e[number][4821],e[number][4822],e[number][4823],e[number][4824],e[number][4825],e[number][4826],e[number][4827],e[number][4828],e[number][4829],e[number][4830],\
e[number][4831],e[number][4832],e[number][4833],e[number][4834],e[number][4835],e[number][4836],e[number][4837],e[number][4838],e[number][4839],e[number][4840],\
e[number][4841],e[number][4842],e[number][4843],e[number][4844],e[number][4845],e[number][4846],e[number][4847],e[number][4848],e[number][4849],e[number][4850],\
e[number][4851],e[number][4852],e[number][4853],e[number][4854],e[number][4855],e[number][4856],e[number][4857],e[number][4858],e[number][4859],e[number][4860],\
e[number][4861],e[number][4862],e[number][4863],e[number][4864],e[number][4865],e[number][4866],e[number][4867],e[number][4868],e[number][4869],e[number][4870],\
e[number][4871],e[number][4872],e[number][4873],e[number][4874],e[number][4875],e[number][4876],e[number][4877],e[number][4878],e[number][4879],e[number][4880],\
e[number][4881],e[number][4882],e[number][4883],e[number][4884],e[number][4885],e[number][4886],e[number][4887],e[number][4888],e[number][4889],e[number][4890],\
e[number][4891],e[number][4892],e[number][4893],e[number][4894],e[number][4895],e[number][4896],e[number][4897],e[number][4898],e[number][4899],e[number][4900],\
e[number][4901],e[number][4902],e[number][4903],e[number][4904],e[number][4905],e[number][4906],e[number][4907],e[number][4908],e[number][4909],e[number][4910],\
e[number][4911],e[number][4912],e[number][4913],e[number][4914],e[number][4915],e[number][4916],e[number][4917],e[number][4918],e[number][4919],e[number][4920],\
e[number][4921],e[number][4922],e[number][4923],e[number][4924],e[number][4925],e[number][4926],e[number][4927],e[number][4928],e[number][4929],e[number][4930],\
e[number][4931],e[number][4932],e[number][4933],e[number][4934],e[number][4935],e[number][4936],e[number][4937],e[number][4938],e[number][4939],e[number][4940],\
e[number][4941],e[number][4942],e[number][4943],e[number][4944],e[number][4945],e[number][4946],e[number][4947],e[number][4948],e[number][4949],e[number][4950],\
e[number][4951],e[number][4952],e[number][4953],e[number][4954],e[number][4955],e[number][4956],e[number][4957],e[number][4958],e[number][4959],e[number][4960],\
e[number][4961],e[number][4962],e[number][4963],e[number][4964],e[number][4965],e[number][4966],e[number][4967],e[number][4968],e[number][4969],e[number][4970],\
e[number][4971],e[number][4972],e[number][4973],e[number][4974],e[number][4975],e[number][4976],e[number][4977],e[number][4978],e[number][4979],e[number][4980],\
e[number][4981],e[number][4982],e[number][4983],e[number][4984],e[number][4985],e[number][4986],e[number][4987],e[number][4988],e[number][4989],e[number][4990],\
e[number][4991],e[number][4992],e[number][4993],e[number][4994],e[number][4995],e[number][4996],e[number][4997],e[number][4998],e[number][4999],e[number][5000],\
e[number][5001],e[number][5002],e[number][5003],e[number][5004],e[number][5005],e[number][5006],e[number][5007],e[number][5008],e[number][5009],e[number][5010],\
e[number][5011],e[number][5012],e[number][5013],e[number][5014],e[number][5015],e[number][5016],e[number][5017],e[number][5018],e[number][5019],e[number][5020],\
e[number][5021],e[number][5022],e[number][5023],e[number][5024],e[number][5025],e[number][5026],e[number][5027],e[number][5028],e[number][5029],e[number][5030],\
e[number][5031],e[number][5032],e[number][5033],e[number][5034],e[number][5035],e[number][5036],e[number][5037],e[number][5038],e[number][5039],e[number][5040],\
e[number][5041],e[number][5042],e[number][5043],e[number][5044],e[number][5045],e[number][5046],e[number][5047],e[number][5048],e[number][5049],e[number][5050],\
e[number][5051],e[number][5052],e[number][5053],e[number][5054],e[number][5055],e[number][5056],e[number][5057],e[number][5058],e[number][5059],e[number][5060],\
e[number][5061],e[number][5062],e[number][5063],e[number][5064],e[number][5065],e[number][5066],e[number][5067],e[number][5068],e[number][5069],e[number][5070],\
e[number][5071],e[number][5072],e[number][5073],e[number][5074],e[number][5075],e[number][5076],e[number][5077],e[number][5078],e[number][5079],e[number][5080],\
e[number][5081],e[number][5082],e[number][5083],e[number][5084],e[number][5085],e[number][5086],e[number][5087],e[number][5088],e[number][5089],e[number][5090],\
e[number][5091],e[number][5092],e[number][5093],e[number][5094],e[number][5095],e[number][5096],e[number][5097],e[number][5098],e[number][5099],e[number][5100],\
e[number][5101],e[number][5102],e[number][5103],e[number][5104],e[number][5105],e[number][5106],e[number][5107],e[number][5108],e[number][5109],e[number][5110],\
e[number][5111],e[number][5112],e[number][5113],e[number][5114],e[number][5115],e[number][5116],e[number][5117],e[number][5118],e[number][5119],e[number][5120],\
e[number][5121],e[number][5122],e[number][5123],e[number][5124],e[number][5125],e[number][5126],e[number][5127],e[number][5128],e[number][5129],e[number][5130],\
e[number][5131],e[number][5132],e[number][5133],e[number][5134],e[number][5135],e[number][5136],e[number][5137],e[number][5138],e[number][5139],e[number][5140],\
e[number][5141],e[number][5142],e[number][5143],e[number][5144],e[number][5145],e[number][5146],e[number][5147],e[number][5148],e[number][5149],e[number][5150],\
e[number][5151],e[number][5152],e[number][5153],e[number][5154],e[number][5155],e[number][5156],e[number][5157],e[number][5158],e[number][5159],e[number][5160],\
e[number][5161],e[number][5162],e[number][5163],e[number][5164],e[number][5165],e[number][5166],e[number][5167],e[number][5168],e[number][5169],e[number][5170],\
e[number][5171],e[number][5172],e[number][5173],e[number][5174],e[number][5175],e[number][5176],e[number][5177],e[number][5178],e[number][5179],e[number][5180],\
e[number][5181],e[number][5182],e[number][5183],e[number][5184],e[number][5185],e[number][5186],e[number][5187],e[number][5188],e[number][5189],e[number][5190],\
e[number][5191],e[number][5192],e[number][5193],e[number][5194],e[number][5195],e[number][5196],e[number][5197],e[number][5198],e[number][5199],e[number][5200],\
e[number][5201],e[number][5202],e[number][5203],e[number][5204],e[number][5205],e[number][5206],e[number][5207],e[number][5208],e[number][5209],e[number][5210],\
e[number][5211],e[number][5212],e[number][5213],e[number][5214],e[number][5215],e[number][5216],e[number][5217],e[number][5218],e[number][5219],e[number][5220],\
e[number][5221],e[number][5222],e[number][5223],e[number][5224],e[number][5225],e[number][5226],e[number][5227],e[number][5228],e[number][5229],e[number][5230],\
e[number][5231],e[number][5232],e[number][5233],e[number][5234],e[number][5235],e[number][5236],e[number][5237],e[number][5238],e[number][5239],e[number][5240],\
e[number][5241],e[number][5242],e[number][5243],e[number][5244],e[number][5245],e[number][5246],e[number][5247],e[number][5248],e[number][5249],e[number][5250],\
e[number][5251],e[number][5252],e[number][5253],e[number][5254],e[number][5255],e[number][5256],e[number][5257],e[number][5258],e[number][5259],e[number][5260],\
e[number][5261],e[number][5262],e[number][5263],e[number][5264],e[number][5265],e[number][5266],e[number][5267],e[number][5268],e[number][5269],e[number][5270],\
e[number][5271],e[number][5272],e[number][5273],e[number][5274],e[number][5275],e[number][5276],e[number][5277],e[number][5278],e[number][5279],e[number][5280],\
e[number][5281],e[number][5282],e[number][5283],e[number][5284],e[number][5285],e[number][5286],e[number][5287],e[number][5288],e[number][5289],e[number][5290],\
e[number][5291],e[number][5292],e[number][5293],e[number][5294],e[number][5295],e[number][5296],e[number][5297],e[number][5298],e[number][5299],e[number][5300],\
e[number][5301],e[number][5302],e[number][5303],e[number][5304],e[number][5305],e[number][5306],e[number][5307],e[number][5308],e[number][5309],e[number][5310],\
e[number][5311],e[number][5312],e[number][5313],e[number][5314],e[number][5315],e[number][5316],e[number][5317],e[number][5318],e[number][5319],e[number][5320],\
e[number][5321],e[number][5322],e[number][5323],e[number][5324],e[number][5325],e[number][5326],e[number][5327],e[number][5328],e[number][5329],e[number][5330],\
e[number][5331],e[number][5332],e[number][5333],e[number][5334],e[number][5335],e[number][5336],e[number][5337],e[number][5338],e[number][5339],e[number][5340],\
e[number][5341],e[number][5342],e[number][5343],e[number][5344],e[number][5345],e[number][5346],e[number][5347],e[number][5348],e[number][5349],e[number][5350],\
e[number][5351],e[number][5352],e[number][5353],e[number][5354],e[number][5355],e[number][5356],e[number][5357],e[number][5358],e[number][5359],e[number][5360],\
e[number][5361],e[number][5362],e[number][5363],e[number][5364],e[number][5365],e[number][5366],e[number][5367],e[number][5368],e[number][5369],e[number][5370],\
e[number][5371],e[number][5372],e[number][5373],e[number][5374],e[number][5375],e[number][5376],e[number][5377],e[number][5378],e[number][5379],e[number][5380],\
e[number][5381],e[number][5382],e[number][5383],e[number][5384],e[number][5385],e[number][5386],e[number][5387],e[number][5388],e[number][5389],e[number][5390],\
e[number][5391],e[number][5392],e[number][5393],e[number][5394],e[number][5395],e[number][5396],e[number][5397],e[number][5398],e[number][5399],e[number][5400],\
e[number][5401],e[number][5402],e[number][5403],e[number][5404],e[number][5405],e[number][5406],e[number][5407],e[number][5408],e[number][5409],e[number][5410],\
e[number][5411],e[number][5412],e[number][5413],e[number][5414],e[number][5415],e[number][5416],e[number][5417],e[number][5418],e[number][5419],e[number][5420],\
e[number][5421],e[number][5422],e[number][5423],e[number][5424],e[number][5425],e[number][5426],e[number][5427],e[number][5428],e[number][5429],e[number][5430],\
e[number][5431],e[number][5432],e[number][5433],e[number][5434],e[number][5435],e[number][5436],e[number][5437],e[number][5438],e[number][5439],e[number][5440],\
e[number][5441],e[number][5442],e[number][5443],e[number][5444],e[number][5445],e[number][5446],e[number][5447],e[number][5448],e[number][5449],e[number][5450],\
e[number][5451],e[number][5452],e[number][5453],e[number][5454],e[number][5455],e[number][5456],e[number][5457],e[number][5458],e[number][5459],e[number][5460],\
e[number][5461],e[number][5462],e[number][5463],e[number][5464],e[number][5465],e[number][5466],e[number][5467],e[number][5468],e[number][5469],e[number][5470],\
e[number][5471],e[number][5472],e[number][5473],e[number][5474],e[number][5475],e[number][5476],e[number][5477],e[number][5478],e[number][5479],e[number][5480],\
e[number][5481],e[number][5482],e[number][5483],e[number][5484],e[number][5485],e[number][5486],e[number][5487],e[number][5488],e[number][5489],e[number][5490],\
e[number][5491],e[number][5492],e[number][5493],e[number][5494],e[number][5495],e[number][5496],e[number][5497],e[number][5498],e[number][5499],e[number][5500],\
e[number][5501],e[number][5502],e[number][5503],e[number][5504],e[number][5505],e[number][5506],e[number][5507],e[number][5508],e[number][5509],e[number][5510],\
e[number][5511],e[number][5512],e[number][5513],e[number][5514],e[number][5515],e[number][5516],e[number][5517],e[number][5518],e[number][5519],e[number][5520],\
e[number][5521],e[number][5522],e[number][5523],e[number][5524],e[number][5525],e[number][5526],e[number][5527],e[number][5528],e[number][5529],e[number][5530],\
e[number][5531],e[number][5532],e[number][5533],e[number][5534],e[number][5535],e[number][5536],e[number][5537],e[number][5538],e[number][5539],e[number][5540],\
e[number][5541],e[number][5542],e[number][5543],e[number][5544],e[number][5545],e[number][5546],e[number][5547],e[number][5548],e[number][5549],e[number][5550],\
e[number][5551],e[number][5552],e[number][5553],e[number][5554],e[number][5555],e[number][5556],e[number][5557],e[number][5558],e[number][5559],e[number][5560],\
e[number][5561],e[number][5562],e[number][5563],e[number][5564],e[number][5565],e[number][5566],e[number][5567],e[number][5568],e[number][5569],e[number][5570],\
e[number][5571],e[number][5572],e[number][5573],e[number][5574],e[number][5575],e[number][5576],e[number][5577],e[number][5578],e[number][5579],e[number][5580],\
e[number][5581],e[number][5582],e[number][5583],e[number][5584],e[number][5585],e[number][5586],e[number][5587],e[number][5588],e[number][5589],e[number][5590],\
e[number][5591],e[number][5592],e[number][5593],e[number][5594],e[number][5595],e[number][5596],e[number][5597],e[number][5598],e[number][5599],e[number][5600],\
e[number][5601],e[number][5602],e[number][5603],e[number][5604],e[number][5605],e[number][5606],e[number][5607],e[number][5608],e[number][5609],e[number][5610],\
e[number][5611],e[number][5612],e[number][5613],e[number][5614],e[number][5615],e[number][5616],e[number][5617],e[number][5618],e[number][5619],e[number][5620],\
e[number][5621],e[number][5622],e[number][5623],e[number][5624],e[number][5625],e[number][5626],e[number][5627],e[number][5628],e[number][5629],e[number][5630],\
e[number][5631],e[number][5632],e[number][5633],e[number][5634],e[number][5635],e[number][5636],e[number][5637],e[number][5638],e[number][5639],e[number][5640],\
e[number][5641],e[number][5642],e[number][5643],e[number][5644],e[number][5645],e[number][5646],e[number][5647],e[number][5648],e[number][5649],e[number][5650],\
e[number][5651],e[number][5652],e[number][5653],e[number][5654],e[number][5655],e[number][5656],e[number][5657],e[number][5658],e[number][5659],e[number][5660],\
e[number][5661],e[number][5662],e[number][5663],e[number][5664],e[number][5665],e[number][5666],e[number][5667],e[number][5668],e[number][5669],e[number][5670],\
e[number][5671],e[number][5672],e[number][5673],e[number][5674],e[number][5675],e[number][5676],e[number][5677],e[number][5678],e[number][5679],e[number][5680],\
e[number][5681],e[number][5682],e[number][5683],e[number][5684],e[number][5685],e[number][5686],e[number][5687],e[number][5688],e[number][5689],e[number][5690],\
e[number][5691],e[number][5692],e[number][5693],e[number][5694],e[number][5695],e[number][5696],e[number][5697],e[number][5698],e[number][5699],e[number][5700],\
e[number][5701],e[number][5702],e[number][5703],e[number][5704],e[number][5705],e[number][5706],e[number][5707],e[number][5708],e[number][5709],e[number][5710],\
e[number][5711],e[number][5712],e[number][5713],e[number][5714],e[number][5715],e[number][5716],e[number][5717],e[number][5718],e[number][5719],e[number][5720],\
e[number][5721],e[number][5722],e[number][5723],e[number][5724],e[number][5725],e[number][5726],e[number][5727],e[number][5728],e[number][5729],e[number][5730],\
e[number][5731],e[number][5732],e[number][5733],e[number][5734],e[number][5735],e[number][5736],e[number][5737],e[number][5738],e[number][5739],e[number][5740],\
e[number][5741],e[number][5742],e[number][5743],e[number][5744],e[number][5745],e[number][5746],e[number][5747],e[number][5748],e[number][5749],e[number][5750],\
e[number][5751],e[number][5752],e[number][5753],e[number][5754],e[number][5755],e[number][5756],e[number][5757],e[number][5758],e[number][5759],e[number][5760],\
e[number][5761],e[number][5762],e[number][5763],e[number][5764],e[number][5765],e[number][5766],e[number][5767],e[number][5768],e[number][5769],e[number][5770],\
e[number][5771],e[number][5772],e[number][5773],e[number][5774],e[number][5775],e[number][5776],e[number][5777],e[number][5778],e[number][5779],e[number][5780],\
e[number][5781],e[number][5782],e[number][5783],e[number][5784],e[number][5785],e[number][5786],e[number][5787],e[number][5788],e[number][5789],e[number][5790],\
e[number][5791],e[number][5792],e[number][5793],e[number][5794],e[number][5795],e[number][5796],e[number][5797],e[number][5798],e[number][5799],e[number][5800],\
e[number][5801],e[number][5802],e[number][5803],e[number][5804],e[number][5805],e[number][5806],e[number][5807],e[number][5808],e[number][5809],e[number][5810],\
e[number][5811],e[number][5812],e[number][5813],e[number][5814],e[number][5815],e[number][5816],e[number][5817],e[number][5818],e[number][5819],e[number][5820],\
e[number][5821],e[number][5822],e[number][5823],e[number][5824],e[number][5825],e[number][5826],e[number][5827],e[number][5828],e[number][5829],e[number][5830],\
e[number][5831],e[number][5832],e[number][5833],e[number][5834],e[number][5835],e[number][5836],e[number][5837],e[number][5838],e[number][5839],e[number][5840],\
e[number][5841],e[number][5842],e[number][5843],e[number][5844],e[number][5845],e[number][5846],e[number][5847],e[number][5848],e[number][5849],e[number][5850],\
e[number][5851],e[number][5852],e[number][5853],e[number][5854],e[number][5855],e[number][5856],e[number][5857],e[number][5858],e[number][5859],e[number][5860],\
e[number][5861],e[number][5862],e[number][5863],e[number][5864],e[number][5865],e[number][5866],e[number][5867],e[number][5868],e[number][5869],e[number][5870],\
e[number][5871],e[number][5872],e[number][5873],e[number][5874],e[number][5875],e[number][5876],e[number][5877],e[number][5878],e[number][5879],e[number][5880],\
e[number][5881],e[number][5882],e[number][5883],e[number][5884],e[number][5885],e[number][5886],e[number][5887],e[number][5888],e[number][5889],e[number][5890],\
e[number][5891],e[number][5892],e[number][5893],e[number][5894],e[number][5895],e[number][5896],e[number][5897],e[number][5898],e[number][5899],e[number][5900],\
e[number][5901],e[number][5902],e[number][5903],e[number][5904],e[number][5905],e[number][5906],e[number][5907],e[number][5908],e[number][5909],e[number][5910],\
e[number][5911],e[number][5912],e[number][5913],e[number][5914],e[number][5915],e[number][5916],e[number][5917],e[number][5918],e[number][5919],e[number][5920],\
e[number][5921],e[number][5922],e[number][5923],e[number][5924],e[number][5925],e[number][5926],e[number][5927],e[number][5928],e[number][5929],e[number][5930],\
e[number][5931],e[number][5932],e[number][5933],e[number][5934],e[number][5935],e[number][5936],e[number][5937],e[number][5938],e[number][5939],e[number][5940],\
e[number][5941],e[number][5942],e[number][5943],e[number][5944],e[number][5945],e[number][5946],e[number][5947],e[number][5948],e[number][5949],e[number][5950],\
e[number][5951],e[number][5952],e[number][5953],e[number][5954],e[number][5955],e[number][5956],e[number][5957],e[number][5958],e[number][5959],e[number][5960],\
e[number][5961],e[number][5962],e[number][5963],e[number][5964],e[number][5965],e[number][5966],e[number][5967],e[number][5968],e[number][5969],e[number][5970],\
e[number][5971],e[number][5972],e[number][5973],e[number][5974],e[number][5975],e[number][5976],e[number][5977],e[number][5978],e[number][5979],e[number][5980],\
e[number][5981],e[number][5982],e[number][5983],e[number][5984],e[number][5985],e[number][5986],e[number][5987],e[number][5988],e[number][5989],e[number][5990],\
e[number][5991],e[number][5992],e[number][5993],e[number][5994],e[number][5995],e[number][5996],e[number][5997],e[number][5998],e[number][5999],e[number][6000],\
e[number][6001],e[number][6002],e[number][6003],e[number][6004],e[number][6005],e[number][6006],e[number][6007],e[number][6008],e[number][6009],e[number][6010],\
e[number][6011],e[number][6012],e[number][6013],e[number][6014],e[number][6015],e[number][6016],e[number][6017],e[number][6018],e[number][6019],e[number][6020],\
e[number][6021],e[number][6022],e[number][6023],e[number][6024],e[number][6025],e[number][6026],e[number][6027],e[number][6028],e[number][6029],e[number][6030],\
e[number][6031],e[number][6032],e[number][6033],e[number][6034],e[number][6035],e[number][6036],e[number][6037],e[number][6038],e[number][6039],e[number][6040],\
e[number][6041],e[number][6042],e[number][6043],e[number][6044],e[number][6045],e[number][6046],e[number][6047],e[number][6048],e[number][6049],e[number][6050],\
e[number][6051],e[number][6052],e[number][6053],e[number][6054],e[number][6055],e[number][6056],e[number][6057],e[number][6058],e[number][6059],e[number][6060],\
e[number][6061],e[number][6062],e[number][6063],e[number][6064],e[number][6065],e[number][6066],e[number][6067],e[number][6068],e[number][6069],e[number][6070],\
e[number][6071],e[number][6072],e[number][6073],e[number][6074],e[number][6075],e[number][6076],e[number][6077],e[number][6078],e[number][6079],e[number][6080],\
e[number][6081],e[number][6082],e[number][6083],e[number][6084],e[number][6085],e[number][6086],e[number][6087],e[number][6088],e[number][6089],e[number][6090],\
e[number][6091],e[number][6092],e[number][6093],e[number][6094],e[number][6095],e[number][6096],e[number][6097],e[number][6098],e[number][6099],e[number][6100],\
e[number][6101],e[number][6102],e[number][6103],e[number][6104],e[number][6105],e[number][6106],e[number][6107],e[number][6108],e[number][6109],e[number][6110],\
e[number][6111],e[number][6112],e[number][6113],e[number][6114],e[number][6115],e[number][6116],e[number][6117],e[number][6118],e[number][6119],e[number][6120],\
e[number][6121],e[number][6122],e[number][6123],e[number][6124],e[number][6125],e[number][6126],e[number][6127],e[number][6128],e[number][6129],e[number][6130],\
e[number][6131],e[number][6132],e[number][6133],e[number][6134],e[number][6135],e[number][6136],e[number][6137],e[number][6138],e[number][6139],e[number][6140],\
e[number][6141],e[number][6142],e[number][6143],e[number][6144],e[number][6145],e[number][6146],e[number][6147],e[number][6148],e[number][6149],e[number][6150],\
e[number][6151],e[number][6152],e[number][6153],e[number][6154],e[number][6155],e[number][6156],e[number][6157],e[number][6158],e[number][6159],e[number][6160],\
e[number][6161],e[number][6162],e[number][6163],e[number][6164],e[number][6165],e[number][6166],e[number][6167],e[number][6168],e[number][6169],e[number][6170],\
e[number][6171],e[number][6172],e[number][6173],e[number][6174],e[number][6175],e[number][6176],e[number][6177],e[number][6178],e[number][6179],e[number][6180],\
e[number][6181],e[number][6182],e[number][6183],e[number][6184],e[number][6185],e[number][6186],e[number][6187],e[number][6188],e[number][6189],e[number][6190],\
e[number][6191],e[number][6192],e[number][6193],e[number][6194],e[number][6195],e[number][6196],e[number][6197],e[number][6198],e[number][6199],e[number][6200],\
e[number][6201],e[number][6202],e[number][6203],e[number][6204],e[number][6205],e[number][6206],e[number][6207],e[number][6208],e[number][6209],e[number][6210],\
e[number][6211],e[number][6212],e[number][6213],e[number][6214],e[number][6215],e[number][6216],e[number][6217],e[number][6218],e[number][6219],e[number][6220],\
e[number][6221],e[number][6222],e[number][6223],e[number][6224],e[number][6225],e[number][6226],e[number][6227],e[number][6228],e[number][6229],e[number][6230],\
e[number][6231],e[number][6232],e[number][6233],e[number][6234],e[number][6235],e[number][6236],e[number][6237],e[number][6238],e[number][6239],e[number][6240],\
e[number][6241],e[number][6242],e[number][6243],e[number][6244],e[number][6245],e[number][6246],e[number][6247],e[number][6248],e[number][6249],e[number][6250],\
e[number][6251],e[number][6252],e[number][6253],e[number][6254],e[number][6255],e[number][6256],e[number][6257],e[number][6258],e[number][6259],e[number][6260],\
e[number][6261],e[number][6262],e[number][6263],e[number][6264],e[number][6265],e[number][6266],e[number][6267],e[number][6268],e[number][6269],e[number][6270],\
e[number][6271],e[number][6272],e[number][6273],e[number][6274],e[number][6275],e[number][6276],e[number][6277],e[number][6278],e[number][6279],e[number][6280],\
e[number][6281],e[number][6282],e[number][6283],e[number][6284],e[number][6285],e[number][6286],e[number][6287],e[number][6288],e[number][6289],e[number][6290],\
e[number][6291],e[number][6292],e[number][6293],e[number][6294],e[number][6295],e[number][6296],e[number][6297],e[number][6298],e[number][6299],e[number][6300],\
e[number][6301],e[number][6302],e[number][6303],e[number][6304],e[number][6305],e[number][6306],e[number][6307],e[number][6308],e[number][6309],e[number][6310],\
e[number][6311],e[number][6312],e[number][6313],e[number][6314],e[number][6315],e[number][6316],e[number][6317],e[number][6318],e[number][6319],e[number][6320],\
e[number][6321],e[number][6322],e[number][6323],e[number][6324],e[number][6325],e[number][6326],e[number][6327],e[number][6328],e[number][6329],e[number][6330],\
e[number][6331],e[number][6332],e[number][6333],e[number][6334],e[number][6335],e[number][6336],e[number][6337],e[number][6338],e[number][6339],e[number][6340],\
e[number][6341],e[number][6342],e[number][6343],e[number][6344],e[number][6345],e[number][6346],e[number][6347],e[number][6348],e[number][6349],e[number][6350],\
e[number][6351],e[number][6352],e[number][6353],e[number][6354],e[number][6355],e[number][6356],e[number][6357],e[number][6358],e[number][6359],e[number][6360],\
e[number][6361],e[number][6362],e[number][6363],e[number][6364],e[number][6365],e[number][6366],e[number][6367],e[number][6368],e[number][6369],e[number][6370],\
e[number][6371],e[number][6372],e[number][6373],e[number][6374],e[number][6375],e[number][6376],e[number][6377],e[number][6378],e[number][6379],e[number][6380],\
e[number][6381],e[number][6382],e[number][6383],e[number][6384],e[number][6385],e[number][6386],e[number][6387],e[number][6388],e[number][6389],e[number][6390],\
e[number][6391],e[number][6392],e[number][6393],e[number][6394],e[number][6395],e[number][6396],e[number][6397],e[number][6398],e[number][6399],e[number][6400],\
e[number][6401],e[number][6402],e[number][6403],e[number][6404],e[number][6405],e[number][6406],e[number][6407],e[number][6408],e[number][6409],e[number][6410],\
e[number][6411],e[number][6412],e[number][6413],e[number][6414],e[number][6415],e[number][6416],e[number][6417],e[number][6418],e[number][6419],e[number][6420],\
e[number][6421],e[number][6422],e[number][6423],e[number][6424],e[number][6425],e[number][6426],e[number][6427],e[number][6428],e[number][6429],e[number][6430],\
e[number][6431],e[number][6432],e[number][6433],e[number][6434],e[number][6435],e[number][6436],e[number][6437],e[number][6438],e[number][6439],e[number][6440],\
e[number][6441],e[number][6442],e[number][6443],e[number][6444],e[number][6445],e[number][6446],e[number][6447],e[number][6448],e[number][6449],e[number][6450],\
e[number][6451],e[number][6452],e[number][6453],e[number][6454],e[number][6455],e[number][6456],e[number][6457],e[number][6458],e[number][6459],e[number][6460],\
e[number][6461],e[number][6462],e[number][6463],e[number][6464],e[number][6465],e[number][6466],e[number][6467],e[number][6468],e[number][6469],e[number][6470],\
e[number][6471],e[number][6472],e[number][6473],e[number][6474],e[number][6475],e[number][6476],e[number][6477],e[number][6478],e[number][6479],e[number][6480],\
e[number][6481],e[number][6482],e[number][6483],e[number][6484],e[number][6485],e[number][6486],e[number][6487],e[number][6488],e[number][6489],e[number][6490],\
e[number][6491],e[number][6492],e[number][6493],e[number][6494],e[number][6495],e[number][6496],e[number][6497],e[number][6498],e[number][6499],e[number][6500],\
e[number][6501],e[number][6502],e[number][6503],e[number][6504],e[number][6505],e[number][6506],e[number][6507],e[number][6508],e[number][6509],e[number][6510],\
e[number][6511],e[number][6512],e[number][6513],e[number][6514],e[number][6515],e[number][6516],e[number][6517],e[number][6518],e[number][6519],e[number][6520],\
e[number][6521],e[number][6522],e[number][6523],e[number][6524],e[number][6525],e[number][6526],e[number][6527],e[number][6528],e[number][6529],e[number][6530],\
e[number][6531],e[number][6532],e[number][6533],e[number][6534],e[number][6535],e[number][6536],e[number][6537],e[number][6538],e[number][6539],e[number][6540],\
e[number][6541],e[number][6542],e[number][6543],e[number][6544],e[number][6545],e[number][6546],e[number][6547],e[number][6548],e[number][6549],e[number][6550],\
e[number][6551],e[number][6552],e[number][6553],e[number][6554],e[number][6555],e[number][6556],e[number][6557],e[number][6558],e[number][6559],e[number][6560],\
e[number][6561],e[number][6562],e[number][6563],e[number][6564],e[number][6565],e[number][6566],e[number][6567],e[number][6568],e[number][6569],e[number][6570],\
e[number][6571],e[number][6572],e[number][6573],e[number][6574],e[number][6575],e[number][6576],e[number][6577],e[number][6578],e[number][6579],e[number][6580],\
e[number][6581],e[number][6582],e[number][6583],e[number][6584],e[number][6585],e[number][6586],e[number][6587],e[number][6588],e[number][6589],e[number][6590],\
e[number][6591],e[number][6592],e[number][6593],e[number][6594],e[number][6595],e[number][6596],e[number][6597],e[number][6598],e[number][6599],e[number][6600],\
e[number][6601],e[number][6602],e[number][6603],e[number][6604],e[number][6605],e[number][6606],e[number][6607],e[number][6608],e[number][6609],e[number][6610],\
e[number][6611],e[number][6612],e[number][6613],e[number][6614],e[number][6615],e[number][6616],e[number][6617],e[number][6618],e[number][6619],e[number][6620],\
e[number][6621],e[number][6622],e[number][6623],e[number][6624],e[number][6625],e[number][6626],e[number][6627],e[number][6628],e[number][6629],e[number][6630],\
e[number][6631],e[number][6632],e[number][6633],e[number][6634],e[number][6635],e[number][6636],e[number][6637],e[number][6638],e[number][6639],e[number][6640],\
e[number][6641],e[number][6642],e[number][6643],e[number][6644],e[number][6645],e[number][6646],e[number][6647],e[number][6648],e[number][6649],e[number][6650],\
e[number][6651],e[number][6652],e[number][6653],e[number][6654],e[number][6655],e[number][6656],e[number][6657],e[number][6658],e[number][6659],e[number][6660],\
e[number][6661],e[number][6662],e[number][6663],e[number][6664],e[number][6665],e[number][6666],e[number][6667],e[number][6668],e[number][6669],e[number][6670],\
e[number][6671],e[number][6672],e[number][6673],e[number][6674],e[number][6675],e[number][6676],e[number][6677],e[number][6678],e[number][6679],e[number][6680],\
e[number][6681],e[number][6682],e[number][6683],e[number][6684],e[number][6685],e[number][6686],e[number][6687],e[number][6688],e[number][6689],e[number][6690],\
e[number][6691],e[number][6692],e[number][6693],e[number][6694],e[number][6695],e[number][6696],e[number][6697],e[number][6698],e[number][6699],e[number][6700],\
e[number][6701],e[number][6702],e[number][6703],e[number][6704],e[number][6705],e[number][6706],e[number][6707],e[number][6708],e[number][6709],e[number][6710],\
e[number][6711],e[number][6712],e[number][6713],e[number][6714],e[number][6715],e[number][6716],e[number][6717],e[number][6718],e[number][6719],e[number][6720],\
e[number][6721],e[number][6722],e[number][6723],e[number][6724],e[number][6725],e[number][6726],e[number][6727],e[number][6728],e[number][6729],e[number][6730],\
e[number][6731],e[number][6732],e[number][6733],e[number][6734],e[number][6735],e[number][6736],e[number][6737],e[number][6738],e[number][6739],e[number][6740],\
e[number][6741],e[number][6742],e[number][6743],e[number][6744],e[number][6745],e[number][6746],e[number][6747],e[number][6748],e[number][6749],e[number][6750],\
e[number][6751],e[number][6752],e[number][6753],e[number][6754],e[number][6755],e[number][6756],e[number][6757],e[number][6758],e[number][6759],e[number][6760],\
e[number][6761],e[number][6762],e[number][6763],e[number][6764],e[number][6765],e[number][6766],e[number][6767],e[number][6768],e[number][6769],e[number][6770],\
e[number][6771],e[number][6772],e[number][6773],e[number][6774],e[number][6775],e[number][6776],e[number][6777],e[number][6778],e[number][6779],e[number][6780],\
e[number][6781],e[number][6782],e[number][6783],e[number][6784],e[number][6785],e[number][6786],e[number][6787],e[number][6788],e[number][6789],e[number][6790],\
e[number][6791],e[number][6792],e[number][6793],e[number][6794],e[number][6795],e[number][6796],e[number][6797],e[number][6798],e[number][6799],e[number][6800],\
e[number][6801],e[number][6802],e[number][6803],e[number][6804],e[number][6805],e[number][6806],e[number][6807],e[number][6808],e[number][6809],e[number][6810],\
e[number][6811],e[number][6812],e[number][6813],e[number][6814],e[number][6815],e[number][6816],e[number][6817],e[number][6818],e[number][6819],e[number][6820],\
e[number][6821],e[number][6822],e[number][6823],e[number][6824],e[number][6825],e[number][6826],e[number][6827],e[number][6828],e[number][6829],e[number][6830],\
e[number][6831],e[number][6832],e[number][6833],e[number][6834],e[number][6835],e[number][6836],e[number][6837],e[number][6838],e[number][6839],e[number][6840],\
e[number][6841],e[number][6842],e[number][6843],e[number][6844],e[number][6845],e[number][6846],e[number][6847],e[number][6848],e[number][6849],e[number][6850],\
e[number][6851],e[number][6852],e[number][6853],e[number][6854],e[number][6855],e[number][6856],e[number][6857],e[number][6858],e[number][6859],e[number][6860],\
e[number][6861],e[number][6862],e[number][6863],e[number][6864],e[number][6865],e[number][6866],e[number][6867],e[number][6868],e[number][6869],e[number][6870],\
e[number][6871],e[number][6872],e[number][6873],e[number][6874],e[number][6875],e[number][6876],e[number][6877],e[number][6878],e[number][6879],e[number][6880],\
e[number][6881],e[number][6882],e[number][6883],e[number][6884],e[number][6885],e[number][6886],e[number][6887],e[number][6888],e[number][6889],e[number][6890],\
e[number][6891],e[number][6892],e[number][6893],e[number][6894],e[number][6895],e[number][6896],e[number][6897],e[number][6898],e[number][6899],e[number][6900],\
e[number][6901],e[number][6902],e[number][6903],e[number][6904],e[number][6905],e[number][6906],e[number][6907],e[number][6908],e[number][6909],e[number][6910],\
e[number][6911],e[number][6912],e[number][6913],e[number][6914],e[number][6915],e[number][6916],e[number][6917],e[number][6918],e[number][6919],e[number][6920],\
e[number][6921],e[number][6922],e[number][6923],e[number][6924],e[number][6925],e[number][6926],e[number][6927],e[number][6928],e[number][6929],e[number][6930],\
e[number][6931],e[number][6932],e[number][6933],e[number][6934],e[number][6935],e[number][6936],e[number][6937],e[number][6938],e[number][6939],e[number][6940],\
e[number][6941],e[number][6942],e[number][6943],e[number][6944],e[number][6945],e[number][6946],e[number][6947],e[number][6948],e[number][6949],e[number][6950],\
e[number][6951],e[number][6952],e[number][6953],e[number][6954],e[number][6955],e[number][6956],e[number][6957],e[number][6958],e[number][6959],e[number][6960],\
e[number][6961],e[number][6962],e[number][6963],e[number][6964],e[number][6965],e[number][6966],e[number][6967],e[number][6968],e[number][6969],e[number][6970],\
e[number][6971],e[number][6972],e[number][6973],e[number][6974],e[number][6975],e[number][6976],e[number][6977],e[number][6978],e[number][6979],e[number][6980],\
e[number][6981],e[number][6982],e[number][6983],e[number][6984],e[number][6985],e[number][6986],e[number][6987],e[number][6988],e[number][6989],e[number][6990],\
e[number][6991],e[number][6992],e[number][6993],e[number][6994],e[number][6995],e[number][6996],e[number][6997],e[number][6998],e[number][6999],e[number][7000],\
e[number][7001],e[number][7002],e[number][7003],e[number][7004],e[number][7005],e[number][7006],e[number][7007],e[number][7008],e[number][7009],e[number][7010],\
e[number][7011],e[number][7012],e[number][7013],e[number][7014],e[number][7015],e[number][7016],e[number][7017],e[number][7018],e[number][7019],e[number][7020],\
e[number][7021],e[number][7022],e[number][7023],e[number][7024],e[number][7025],e[number][7026],e[number][7027],e[number][7028],e[number][7029],e[number][7030],\
e[number][7031],e[number][7032],e[number][7033],e[number][7034],e[number][7035],e[number][7036],e[number][7037],e[number][7038],e[number][7039],e[number][7040],\
e[number][7041],e[number][7042],e[number][7043],e[number][7044],e[number][7045],e[number][7046],e[number][7047],e[number][7048],e[number][7049],e[number][7050],\
e[number][7051],e[number][7052],e[number][7053],e[number][7054],e[number][7055],e[number][7056],e[number][7057],e[number][7058],e[number][7059],e[number][7060],\
e[number][7061],e[number][7062],e[number][7063],e[number][7064],e[number][7065],e[number][7066],e[number][7067],e[number][7068],e[number][7069],e[number][7070],\
e[number][7071],e[number][7072],e[number][7073],e[number][7074],e[number][7075],e[number][7076],e[number][7077],e[number][7078],e[number][7079],e[number][7080],\
e[number][7081],e[number][7082],e[number][7083],e[number][7084],e[number][7085],e[number][7086],e[number][7087],e[number][7088],e[number][7089],e[number][7090],\
e[number][7091],e[number][7092],e[number][7093],e[number][7094],e[number][7095],e[number][7096],e[number][7097],e[number][7098],e[number][7099],e[number][7100],\
e[number][7101],e[number][7102],e[number][7103],e[number][7104],e[number][7105],e[number][7106],e[number][7107],e[number][7108],e[number][7109],e[number][7110],\
e[number][7111],e[number][7112],e[number][7113],e[number][7114],e[number][7115],e[number][7116],e[number][7117],e[number][7118],e[number][7119],e[number][7120],\
e[number][7121],e[number][7122],e[number][7123],e[number][7124],e[number][7125],e[number][7126],e[number][7127],e[number][7128],e[number][7129],e[number][7130],\
e[number][7131],e[number][7132],e[number][7133],e[number][7134],e[number][7135],e[number][7136],e[number][7137],e[number][7138],e[number][7139],e[number][7140],\
e[number][7141],e[number][7142],e[number][7143],e[number][7144],e[number][7145],e[number][7146],e[number][7147],e[number][7148],e[number][7149],e[number][7150],\
e[number][7151],e[number][7152],e[number][7153],e[number][7154],e[number][7155],e[number][7156],e[number][7157],e[number][7158],e[number][7159],e[number][7160],\
e[number][7161],e[number][7162],e[number][7163],e[number][7164],e[number][7165],e[number][7166],e[number][7167],e[number][7168],e[number][7169],e[number][7170],\
e[number][7171],e[number][7172],e[number][7173],e[number][7174],e[number][7175],e[number][7176],e[number][7177],e[number][7178],e[number][7179],e[number][7180],\
e[number][7181],e[number][7182],e[number][7183],e[number][7184],e[number][7185],e[number][7186],e[number][7187],e[number][7188],e[number][7189],e[number][7190],\
e[number][7191],e[number][7192],e[number][7193],e[number][7194],e[number][7195],e[number][7196],e[number][7197],e[number][7198],e[number][7199],e[number][7200],\
e[number][7201],e[number][7202],e[number][7203],e[number][7204],e[number][7205],e[number][7206],e[number][7207],e[number][7208],e[number][7209],e[number][7210],\
e[number][7211],e[number][7212],e[number][7213],e[number][7214],e[number][7215],e[number][7216],e[number][7217],e[number][7218],e[number][7219],e[number][7220],\
e[number][7221],e[number][7222],e[number][7223],e[number][7224],e[number][7225],e[number][7226],e[number][7227],e[number][7228],e[number][7229],e[number][7230],\
e[number][7231],e[number][7232],e[number][7233],e[number][7234],e[number][7235],e[number][7236],e[number][7237],e[number][7238],e[number][7239],e[number][7240],\
e[number][7241],e[number][7242],e[number][7243],e[number][7244],e[number][7245],e[number][7246],e[number][7247],e[number][7248],e[number][7249],e[number][7250],\
e[number][7251],e[number][7252],e[number][7253],e[number][7254],e[number][7255],e[number][7256],e[number][7257],e[number][7258],e[number][7259],e[number][7260],\
e[number][7261],e[number][7262],e[number][7263],e[number][7264],e[number][7265],e[number][7266],e[number][7267],e[number][7268],e[number][7269],e[number][7270],\
e[number][7271],e[number][7272],e[number][7273],e[number][7274],e[number][7275],e[number][7276],e[number][7277],e[number][7278],e[number][7279],e[number][7280],\
e[number][7281],e[number][7282],e[number][7283],e[number][7284],e[number][7285],e[number][7286],e[number][7287],e[number][7288],e[number][7289],e[number][7290],\
e[number][7291],e[number][7292],e[number][7293],e[number][7294],e[number][7295],e[number][7296],e[number][7297],e[number][7298],e[number][7299],e[number][7300],\
e[number][7301],e[number][7302],e[number][7303],e[number][7304],e[number][7305],e[number][7306],e[number][7307],e[number][7308],e[number][7309],e[number][7310],\
e[number][7311],e[number][7312],e[number][7313],e[number][7314],e[number][7315],e[number][7316],e[number][7317],e[number][7318],e[number][7319],e[number][7320],\
e[number][7321],e[number][7322],e[number][7323],e[number][7324],e[number][7325],e[number][7326],e[number][7327],e[number][7328],e[number][7329],e[number][7330],\
e[number][7331],e[number][7332],e[number][7333],e[number][7334],e[number][7335],e[number][7336],e[number][7337],e[number][7338],e[number][7339],e[number][7340],\
e[number][7341],e[number][7342],e[number][7343],e[number][7344],e[number][7345],e[number][7346],e[number][7347],e[number][7348],e[number][7349],e[number][7350],\
e[number][7351],e[number][7352],e[number][7353],e[number][7354],e[number][7355],e[number][7356],e[number][7357],e[number][7358],e[number][7359],e[number][7360],\
e[number][7361],e[number][7362],e[number][7363],e[number][7364],e[number][7365],e[number][7366],e[number][7367],e[number][7368],e[number][7369],e[number][7370],\
e[number][7371],e[number][7372],e[number][7373],e[number][7374],e[number][7375],e[number][7376],e[number][7377],e[number][7378],e[number][7379],e[number][7380],\
e[number][7381],e[number][7382],e[number][7383],e[number][7384],e[number][7385],e[number][7386],e[number][7387],e[number][7388],e[number][7389],e[number][7390],\
e[number][7391],e[number][7392],e[number][7393],e[number][7394],e[number][7395],e[number][7396],e[number][7397],e[number][7398],e[number][7399],e[number][7400],\
e[number][7401],e[number][7402],e[number][7403],e[number][7404],e[number][7405],e[number][7406],e[number][7407],e[number][7408],e[number][7409],e[number][7410],\
e[number][7411],e[number][7412],e[number][7413],e[number][7414],e[number][7415],e[number][7416],e[number][7417],e[number][7418],e[number][7419],e[number][7420],\
e[number][7421],e[number][7422],e[number][7423],e[number][7424],e[number][7425],e[number][7426],e[number][7427],e[number][7428],e[number][7429],e[number][7430],\
e[number][7431],e[number][7432],e[number][7433],e[number][7434],e[number][7435],e[number][7436],e[number][7437],e[number][7438],e[number][7439],e[number][7440],\
e[number][7441],e[number][7442],e[number][7443],e[number][7444],e[number][7445],e[number][7446],e[number][7447],e[number][7448],e[number][7449],e[number][7450],\
e[number][7451],e[number][7452],e[number][7453],e[number][7454],e[number][7455],e[number][7456],e[number][7457],e[number][7458],e[number][7459],e[number][7460],\
e[number][7461],e[number][7462],e[number][7463],e[number][7464],e[number][7465],e[number][7466],e[number][7467],e[number][7468],e[number][7469],e[number][7470],\
e[number][7471],e[number][7472],e[number][7473],e[number][7474],e[number][7475],e[number][7476],e[number][7477],e[number][7478],e[number][7479],e[number][7480],\
e[number][7481],e[number][7482],e[number][7483],e[number][7484],e[number][7485],e[number][7486],e[number][7487],e[number][7488],e[number][7489],e[number][7490],\
e[number][7491],e[number][7492],e[number][7493],e[number][7494],e[number][7495],e[number][7496],e[number][7497],e[number][7498],e[number][7499],e[number][7500],\
e[number][7501],e[number][7502],e[number][7503],e[number][7504],e[number][7505],e[number][7506],e[number][7507],e[number][7508],e[number][7509],e[number][7510],\
e[number][7511],e[number][7512],e[number][7513],e[number][7514],e[number][7515],e[number][7516],e[number][7517],e[number][7518],e[number][7519],e[number][7520],\
e[number][7521],e[number][7522],e[number][7523],e[number][7524],e[number][7525],e[number][7526],e[number][7527],e[number][7528],e[number][7529],e[number][7530],\
e[number][7531],e[number][7532],e[number][7533],e[number][7534],e[number][7535],e[number][7536],e[number][7537],e[number][7538],e[number][7539],e[number][7540],\
e[number][7541],e[number][7542],e[number][7543],e[number][7544],e[number][7545],e[number][7546],e[number][7547],e[number][7548],e[number][7549],e[number][7550],\
e[number][7551],e[number][7552],e[number][7553],e[number][7554],e[number][7555],e[number][7556],e[number][7557],e[number][7558],e[number][7559],e[number][7560],\
e[number][7561],e[number][7562],e[number][7563],e[number][7564],e[number][7565],e[number][7566],e[number][7567],e[number][7568],e[number][7569],e[number][7570],\
e[number][7571],e[number][7572],e[number][7573],e[number][7574],e[number][7575],e[number][7576],e[number][7577],e[number][7578],e[number][7579],e[number][7580],\
e[number][7581],e[number][7582],e[number][7583],e[number][7584],e[number][7585],e[number][7586],e[number][7587],e[number][7588],e[number][7589],e[number][7590],\
e[number][7591],e[number][7592],e[number][7593],e[number][7594],e[number][7595],e[number][7596],e[number][7597],e[number][7598],e[number][7599],e[number][7600],\
e[number][7601],e[number][7602],e[number][7603],e[number][7604],e[number][7605],e[number][7606],e[number][7607],e[number][7608],e[number][7609],e[number][7610],\
e[number][7611],e[number][7612],e[number][7613],e[number][7614],e[number][7615],e[number][7616],e[number][7617],e[number][7618],e[number][7619],e[number][7620],\
e[number][7621],e[number][7622],e[number][7623],e[number][7624],e[number][7625],e[number][7626],e[number][7627],e[number][7628],e[number][7629],e[number][7630],\
e[number][7631],e[number][7632],e[number][7633],e[number][7634],e[number][7635],e[number][7636],e[number][7637],e[number][7638],e[number][7639],e[number][7640],\
e[number][7641],e[number][7642],e[number][7643],e[number][7644],e[number][7645],e[number][7646],e[number][7647],e[number][7648],e[number][7649],e[number][7650],\
e[number][7651],e[number][7652],e[number][7653],e[number][7654],e[number][7655],e[number][7656],e[number][7657],e[number][7658],e[number][7659],e[number][7660],\
e[number][7661],e[number][7662],e[number][7663],e[number][7664],e[number][7665],e[number][7666],e[number][7667],e[number][7668],e[number][7669],e[number][7670],\
e[number][7671],e[number][7672],e[number][7673],e[number][7674],e[number][7675],e[number][7676],e[number][7677],e[number][7678],e[number][7679],e[number][7680],\
e[number][7681],e[number][7682],e[number][7683],e[number][7684],e[number][7685],e[number][7686],e[number][7687],e[number][7688],e[number][7689],e[number][7690],\
e[number][7691],e[number][7692],e[number][7693],e[number][7694],e[number][7695],e[number][7696],e[number][7697],e[number][7698],e[number][7699],e[number][7700],\
e[number][7701],e[number][7702],e[number][7703],e[number][7704],e[number][7705],e[number][7706],e[number][7707],e[number][7708],e[number][7709],e[number][7710],\
e[number][7711],e[number][7712],e[number][7713],e[number][7714],e[number][7715],e[number][7716],e[number][7717],e[number][7718],e[number][7719],e[number][7720],\
e[number][7721],e[number][7722],e[number][7723],e[number][7724],e[number][7725],e[number][7726],e[number][7727],e[number][7728],e[number][7729],e[number][7730],\
e[number][7731],e[number][7732],e[number][7733],e[number][7734],e[number][7735],e[number][7736],e[number][7737],e[number][7738],e[number][7739],e[number][7740],\
e[number][7741],e[number][7742],e[number][7743],e[number][7744],e[number][7745],e[number][7746],e[number][7747],e[number][7748],e[number][7749],e[number][7750],\
e[number][7751],e[number][7752],e[number][7753],e[number][7754],e[number][7755],e[number][7756],e[number][7757],e[number][7758],e[number][7759],e[number][7760],\
e[number][7761],e[number][7762],e[number][7763],e[number][7764],e[number][7765],e[number][7766],e[number][7767],e[number][7768],e[number][7769],e[number][7770],\
e[number][7771],e[number][7772],e[number][7773],e[number][7774],e[number][7775],e[number][7776],e[number][7777],e[number][7778],e[number][7779],e[number][7780],\
e[number][7781],e[number][7782],e[number][7783],e[number][7784],e[number][7785],e[number][7786],e[number][7787],e[number][7788],e[number][7789],e[number][7790],\
e[number][7791],e[number][7792],e[number][7793],e[number][7794],e[number][7795],e[number][7796],e[number][7797],e[number][7798],e[number][7799],e[number][7800],\
e[number][7801],e[number][7802],e[number][7803],e[number][7804],e[number][7805],e[number][7806],e[number][7807],e[number][7808],e[number][7809],e[number][7810],\
e[number][7811],e[number][7812],e[number][7813],e[number][7814],e[number][7815],e[number][7816],e[number][7817],e[number][7818],e[number][7819],e[number][7820],\
e[number][7821],e[number][7822],e[number][7823],e[number][7824],e[number][7825],e[number][7826],e[number][7827],e[number][7828],e[number][7829],e[number][7830],\
e[number][7831],e[number][7832],e[number][7833],e[number][7834],e[number][7835],e[number][7836],e[number][7837],e[number][7838],e[number][7839],e[number][7840],\
e[number][7841],e[number][7842],e[number][7843],e[number][7844],e[number][7845],e[number][7846],e[number][7847],e[number][7848],e[number][7849],e[number][7850],\
e[number][7851],e[number][7852],e[number][7853],e[number][7854],e[number][7855],e[number][7856],e[number][7857],e[number][7858],e[number][7859],e[number][7860],\
e[number][7861],e[number][7862],e[number][7863],e[number][7864],e[number][7865],e[number][7866],e[number][7867],e[number][7868],e[number][7869],e[number][7870],\
e[number][7871],e[number][7872],e[number][7873],e[number][7874],e[number][7875],e[number][7876],e[number][7877],e[number][7878],e[number][7879],e[number][7880],\
e[number][7881],e[number][7882],e[number][7883],e[number][7884],e[number][7885],e[number][7886],e[number][7887],e[number][7888],e[number][7889],e[number][7890],\
e[number][7891],e[number][7892],e[number][7893],e[number][7894],e[number][7895],e[number][7896],e[number][7897],e[number][7898],e[number][7899],e[number][7900],\
e[number][7901],e[number][7902],e[number][7903],e[number][7904],e[number][7905],e[number][7906],e[number][7907],e[number][7908],e[number][7909],e[number][7910],\
e[number][7911],e[number][7912],e[number][7913],e[number][7914],e[number][7915],e[number][7916],e[number][7917],e[number][7918],e[number][7919],e[number][7920],\
e[number][7921],e[number][7922],e[number][7923],e[number][7924],e[number][7925],e[number][7926],e[number][7927],e[number][7928],e[number][7929],e[number][7930],\
e[number][7931],e[number][7932],e[number][7933],e[number][7934],e[number][7935],e[number][7936],e[number][7937],e[number][7938],e[number][7939],e[number][7940],\
e[number][7941],e[number][7942],e[number][7943],e[number][7944],e[number][7945],e[number][7946],e[number][7947],e[number][7948],e[number][7949],e[number][7950],\
e[number][7951],e[number][7952],e[number][7953],e[number][7954],e[number][7955],e[number][7956],e[number][7957],e[number][7958],e[number][7959],e[number][7960],\
e[number][7961],e[number][7962],e[number][7963],e[number][7964],e[number][7965],e[number][7966],e[number][7967],e[number][7968],e[number][7969],e[number][7970],\
e[number][7971],e[number][7972],e[number][7973],e[number][7974],e[number][7975],e[number][7976],e[number][7977],e[number][7978],e[number][7979],e[number][7980],\
e[number][7981],e[number][7982],e[number][7983],e[number][7984],e[number][7985],e[number][7986],e[number][7987],e[number][7988],e[number][7989],e[number][7990],\
e[number][7991],e[number][7992],e[number][7993],e[number][7994],e[number][7995],e[number][7996],e[number][7997],e[number][7998],e[number][7999],e[number][8000],\
e[number][8001],e[number][8002],e[number][8003],e[number][8004],e[number][8005],e[number][8006],e[number][8007],e[number][8008],e[number][8009],e[number][8010],\
e[number][8011],e[number][8012],e[number][8013],e[number][8014],e[number][8015],e[number][8016],e[number][8017],e[number][8018],e[number][8019],e[number][8020],\
e[number][8021],e[number][8022],e[number][8023],e[number][8024],e[number][8025],e[number][8026],e[number][8027],e[number][8028],e[number][8029],e[number][8030],\
e[number][8031],e[number][8032],e[number][8033],e[number][8034],e[number][8035],e[number][8036],e[number][8037],e[number][8038],e[number][8039],e[number][8040],\
e[number][8041],e[number][8042],e[number][8043],e[number][8044],e[number][8045],e[number][8046],e[number][8047],e[number][8048],e[number][8049],e[number][8050],\
e[number][8051],e[number][8052],e[number][8053],e[number][8054],e[number][8055],e[number][8056],e[number][8057],e[number][8058],e[number][8059],e[number][8060],\
e[number][8061],e[number][8062],e[number][8063],e[number][8064],e[number][8065],e[number][8066],e[number][8067],e[number][8068],e[number][8069],e[number][8070],\
e[number][8071],e[number][8072],e[number][8073],e[number][8074],e[number][8075],e[number][8076],e[number][8077],e[number][8078],e[number][8079],e[number][8080],\
e[number][8081],e[number][8082],e[number][8083],e[number][8084],e[number][8085],e[number][8086],e[number][8087],e[number][8088],e[number][8089],e[number][8090],\
e[number][8091],e[number][8092],e[number][8093],e[number][8094],e[number][8095],e[number][8096],e[number][8097],e[number][8098],e[number][8099],e[number][8100],\
e[number][8101],e[number][8102],e[number][8103],e[number][8104],e[number][8105],e[number][8106],e[number][8107],e[number][8108],e[number][8109],e[number][8110],\
e[number][8111],e[number][8112],e[number][8113],e[number][8114],e[number][8115],e[number][8116],e[number][8117],e[number][8118],e[number][8119],e[number][8120],\
e[number][8121],e[number][8122],e[number][8123],e[number][8124],e[number][8125],e[number][8126],e[number][8127],e[number][8128],e[number][8129],e[number][8130],\
e[number][8131],e[number][8132],e[number][8133],e[number][8134],e[number][8135],e[number][8136],e[number][8137],e[number][8138],e[number][8139],e[number][8140],\
e[number][8141],e[number][8142],e[number][8143],e[number][8144],e[number][8145],e[number][8146],e[number][8147],e[number][8148],e[number][8149],e[number][8150],\
e[number][8151],e[number][8152],e[number][8153],e[number][8154],e[number][8155],e[number][8156],e[number][8157],e[number][8158],e[number][8159],e[number][8160],\
e[number][8161],e[number][8162],e[number][8163],e[number][8164],e[number][8165],e[number][8166],e[number][8167],e[number][8168],e[number][8169],e[number][8170],\
e[number][8171],e[number][8172],e[number][8173],e[number][8174],e[number][8175],e[number][8176],e[number][8177],e[number][8178],e[number][8179],e[number][8180],\
e[number][8181],e[number][8182],e[number][8183],e[number][8184],e[number][8185],e[number][8186],e[number][8187],e[number][8188],e[number][8189],e[number][8190],\
e[number][8191],e[number][8192],e[number][8193],e[number][8194],e[number][8195],e[number][8196],e[number][8197],e[number][8198],e[number][8199],e[number][8200],\
e[number][8201],e[number][8202],e[number][8203],e[number][8204],e[number][8205],e[number][8206],e[number][8207],e[number][8208],e[number][8209],e[number][8210],\
e[number][8211],e[number][8212],e[number][8213],e[number][8214],e[number][8215],e[number][8216],e[number][8217],e[number][8218],e[number][8219],e[number][8220],\
e[number][8221],e[number][8222],e[number][8223],e[number][8224],e[number][8225],e[number][8226],e[number][8227],e[number][8228],e[number][8229],e[number][8230],\
e[number][8231],e[number][8232],e[number][8233],e[number][8234],e[number][8235],e[number][8236],e[number][8237],e[number][8238],e[number][8239],e[number][8240],\
e[number][8241],e[number][8242],e[number][8243],e[number][8244],e[number][8245],e[number][8246],e[number][8247],e[number][8248],e[number][8249],e[number][8250],\
e[number][8251],e[number][8252],e[number][8253],e[number][8254],e[number][8255],e[number][8256],e[number][8257],e[number][8258],e[number][8259],e[number][8260],\
e[number][8261],e[number][8262],e[number][8263],e[number][8264],e[number][8265],e[number][8266],e[number][8267],e[number][8268],e[number][8269],e[number][8270],\
e[number][8271],e[number][8272],e[number][8273],e[number][8274],e[number][8275],e[number][8276],e[number][8277],e[number][8278],e[number][8279],e[number][8280],\
e[number][8281],e[number][8282],e[number][8283],e[number][8284],e[number][8285],e[number][8286],e[number][8287],e[number][8288],e[number][8289],e[number][8290],\
e[number][8291],e[number][8292],e[number][8293],e[number][8294],e[number][8295],e[number][8296],e[number][8297],e[number][8298],e[number][8299],e[number][8300],\
e[number][8301],e[number][8302],e[number][8303],e[number][8304],e[number][8305],e[number][8306],e[number][8307],e[number][8308],e[number][8309],e[number][8310],\
e[number][8311],e[number][8312],e[number][8313],e[number][8314],e[number][8315],e[number][8316],e[number][8317],e[number][8318],e[number][8319],e[number][8320],\
e[number][8321],e[number][8322],e[number][8323],e[number][8324],e[number][8325],e[number][8326],e[number][8327],e[number][8328],e[number][8329],e[number][8330],\
e[number][8331],e[number][8332],e[number][8333],e[number][8334],e[number][8335],e[number][8336],e[number][8337],e[number][8338],e[number][8339],e[number][8340],\
e[number][8341],e[number][8342],e[number][8343],e[number][8344],e[number][8345],e[number][8346],e[number][8347],e[number][8348],e[number][8349],e[number][8350],\
e[number][8351],e[number][8352],e[number][8353],e[number][8354],e[number][8355],e[number][8356],e[number][8357],e[number][8358],e[number][8359],e[number][8360],\
e[number][8361],e[number][8362],e[number][8363],e[number][8364],e[number][8365],e[number][8366],e[number][8367],e[number][8368],e[number][8369],e[number][8370],\
e[number][8371],e[number][8372],e[number][8373],e[number][8374],e[number][8375],e[number][8376],e[number][8377],e[number][8378],e[number][8379],e[number][8380],\
e[number][8381],e[number][8382],e[number][8383],e[number][8384],e[number][8385],e[number][8386],e[number][8387],e[number][8388],e[number][8389],e[number][8390],\
e[number][8391],e[number][8392],e[number][8393],e[number][8394],e[number][8395],e[number][8396],e[number][8397],e[number][8398],e[number][8399],e[number][8400],\
e[number][8401],e[number][8402],e[number][8403],e[number][8404],e[number][8405],e[number][8406],e[number][8407],e[number][8408],e[number][8409],e[number][8410],\
e[number][8411],e[number][8412],e[number][8413],e[number][8414],e[number][8415],e[number][8416],e[number][8417],e[number][8418],e[number][8419],e[number][8420],\
e[number][8421],e[number][8422],e[number][8423],e[number][8424],e[number][8425],e[number][8426],e[number][8427],e[number][8428],e[number][8429],e[number][8430],\
e[number][8431],e[number][8432],e[number][8433],e[number][8434],e[number][8435],e[number][8436],e[number][8437],e[number][8438],e[number][8439],e[number][8440],\
e[number][8441],e[number][8442],e[number][8443],e[number][8444],e[number][8445],e[number][8446],e[number][8447],e[number][8448],e[number][8449],e[number][8450],\
e[number][8451],e[number][8452],e[number][8453],e[number][8454],e[number][8455],e[number][8456],e[number][8457],e[number][8458],e[number][8459],e[number][8460],\
e[number][8461],e[number][8462],e[number][8463],e[number][8464],e[number][8465],e[number][8466],e[number][8467],e[number][8468],e[number][8469],e[number][8470],\
e[number][8471],e[number][8472],e[number][8473],e[number][8474],e[number][8475],e[number][8476],e[number][8477],e[number][8478],e[number][8479],e[number][8480],\
e[number][8481],e[number][8482],e[number][8483],e[number][8484],e[number][8485],e[number][8486],e[number][8487],e[number][8488],e[number][8489],e[number][8490],\
e[number][8491],e[number][8492],e[number][8493],e[number][8494],e[number][8495],e[number][8496],e[number][8497],e[number][8498],e[number][8499],e[number][8500],\
e[number][8501],e[number][8502],e[number][8503],e[number][8504],e[number][8505],e[number][8506],e[number][8507],e[number][8508],e[number][8509],e[number][8510],\
e[number][8511],e[number][8512],e[number][8513],e[number][8514],e[number][8515],e[number][8516],e[number][8517],e[number][8518],e[number][8519],e[number][8520],\
e[number][8521],e[number][8522],e[number][8523],e[number][8524],e[number][8525],e[number][8526],e[number][8527],e[number][8528],e[number][8529],e[number][8530],\
e[number][8531],e[number][8532],e[number][8533],e[number][8534],e[number][8535],e[number][8536],e[number][8537],e[number][8538],e[number][8539],e[number][8540],\
e[number][8541],e[number][8542],e[number][8543],e[number][8544],e[number][8545],e[number][8546],e[number][8547],e[number][8548],e[number][8549],e[number][8550],\
e[number][8551],e[number][8552],e[number][8553],e[number][8554],e[number][8555],e[number][8556],e[number][8557],e[number][8558],e[number][8559],e[number][8560],\
e[number][8561],e[number][8562],e[number][8563],e[number][8564],e[number][8565],e[number][8566],e[number][8567],e[number][8568],e[number][8569],e[number][8570],\
e[number][8571],e[number][8572],e[number][8573],e[number][8574],e[number][8575],e[number][8576],e[number][8577],e[number][8578],e[number][8579],e[number][8580],\
e[number][8581],e[number][8582],e[number][8583],e[number][8584],e[number][8585],e[number][8586],e[number][8587],e[number][8588],e[number][8589],e[number][8590],\
e[number][8591],e[number][8592],e[number][8593],e[number][8594],e[number][8595],e[number][8596],e[number][8597],e[number][8598],e[number][8599],e[number][8600],\
e[number][8601],e[number][8602],e[number][8603],e[number][8604],e[number][8605],e[number][8606],e[number][8607],e[number][8608],e[number][8609],e[number][8610],\
e[number][8611],e[number][8612],e[number][8613],e[number][8614],e[number][8615],e[number][8616],e[number][8617],e[number][8618],e[number][8619],e[number][8620],\
e[number][8621],e[number][8622],e[number][8623],e[number][8624],e[number][8625],e[number][8626],e[number][8627],e[number][8628],e[number][8629],e[number][8630],\
e[number][8631],e[number][8632],e[number][8633],e[number][8634],e[number][8635],e[number][8636],e[number][8637],e[number][8638],e[number][8639],e[number][8640],\
e[number][8641],e[number][8642],e[number][8643],e[number][8644],e[number][8645],e[number][8646],e[number][8647],e[number][8648],e[number][8649],e[number][8650],\
e[number][8651],e[number][8652],e[number][8653],e[number][8654],e[number][8655],e[number][8656],e[number][8657],e[number][8658],e[number][8659],e[number][8660],\
e[number][8661],e[number][8662],e[number][8663],e[number][8664],e[number][8665],e[number][8666],e[number][8667],e[number][8668],e[number][8669],e[number][8670],\
e[number][8671],e[number][8672],e[number][8673],e[number][8674],e[number][8675],e[number][8676],e[number][8677],e[number][8678],e[number][8679],e[number][8680],\
e[number][8681],e[number][8682],e[number][8683],e[number][8684],e[number][8685],e[number][8686],e[number][8687],e[number][8688],e[number][8689],e[number][8690],\
e[number][8691],e[number][8692],e[number][8693],e[number][8694],e[number][8695],e[number][8696],e[number][8697],e[number][8698],e[number][8699],e[number][8700],\
e[number][8701],e[number][8702],e[number][8703],e[number][8704],e[number][8705],e[number][8706],e[number][8707],e[number][8708],e[number][8709],e[number][8710],\
e[number][8711],e[number][8712],e[number][8713],e[number][8714],e[number][8715],e[number][8716],e[number][8717],e[number][8718],e[number][8719],e[number][8720],\
e[number][8721],e[number][8722],e[number][8723],e[number][8724],e[number][8725],e[number][8726],e[number][8727],e[number][8728],e[number][8729],e[number][8730],\
e[number][8731],e[number][8732],e[number][8733],e[number][8734],e[number][8735],e[number][8736],e[number][8737],e[number][8738],e[number][8739],e[number][8740],\
e[number][8741],e[number][8742],e[number][8743],e[number][8744],e[number][8745],e[number][8746],e[number][8747],e[number][8748],e[number][8749],e[number][8750],\
e[number][8751],e[number][8752],e[number][8753],e[number][8754],e[number][8755],e[number][8756],e[number][8757],e[number][8758],e[number][8759],e[number][8760],\
e[number][8761],e[number][8762],e[number][8763],e[number][8764],e[number][8765],e[number][8766],e[number][8767],e[number][8768],e[number][8769],e[number][8770],\
e[number][8771],e[number][8772],e[number][8773],e[number][8774],e[number][8775],e[number][8776],e[number][8777],e[number][8778],e[number][8779],e[number][8780],\
e[number][8781],e[number][8782],e[number][8783],e[number][8784],e[number][8785],e[number][8786],e[number][8787],e[number][8788],e[number][8789],e[number][8790],\
e[number][8791],e[number][8792],e[number][8793],e[number][8794],e[number][8795],e[number][8796],e[number][8797],e[number][8798],e[number][8799],e[number][8800],\
e[number][8801],e[number][8802],e[number][8803],e[number][8804],e[number][8805],e[number][8806],e[number][8807],e[number][8808],e[number][8809],e[number][8810],\
e[number][8811],e[number][8812],e[number][8813],e[number][8814],e[number][8815],e[number][8816],e[number][8817],e[number][8818],e[number][8819],e[number][8820],\
e[number][8821],e[number][8822],e[number][8823],e[number][8824],e[number][8825],e[number][8826],e[number][8827],e[number][8828],e[number][8829],e[number][8830],\
e[number][8831],e[number][8832],e[number][8833],e[number][8834],e[number][8835],e[number][8836],e[number][8837],e[number][8838],e[number][8839],e[number][8840],\
e[number][8841],e[number][8842],e[number][8843],e[number][8844],e[number][8845],e[number][8846],e[number][8847],e[number][8848],e[number][8849],e[number][8850],\
e[number][8851],e[number][8852],e[number][8853],e[number][8854],e[number][8855],e[number][8856],e[number][8857],e[number][8858],e[number][8859],e[number][8860],\
e[number][8861],e[number][8862],e[number][8863],e[number][8864],e[number][8865],e[number][8866],e[number][8867],e[number][8868],e[number][8869],e[number][8870],\
e[number][8871],e[number][8872],e[number][8873],e[number][8874],e[number][8875],e[number][8876],e[number][8877],e[number][8878],e[number][8879],e[number][8880],\
e[number][8881],e[number][8882],e[number][8883],e[number][8884],e[number][8885],e[number][8886],e[number][8887],e[number][8888],e[number][8889],e[number][8890],\
e[number][8891],e[number][8892],e[number][8893],e[number][8894],e[number][8895],e[number][8896],e[number][8897],e[number][8898],e[number][8899],e[number][8900],\
e[number][8901],e[number][8902],e[number][8903],e[number][8904],e[number][8905],e[number][8906],e[number][8907],e[number][8908],e[number][8909],e[number][8910],\
e[number][8911],e[number][8912],e[number][8913],e[number][8914],e[number][8915],e[number][8916],e[number][8917],e[number][8918],e[number][8919],e[number][8920],\
e[number][8921],e[number][8922],e[number][8923],e[number][8924],e[number][8925],e[number][8926],e[number][8927],e[number][8928],e[number][8929],e[number][8930],\
e[number][8931],e[number][8932],e[number][8933],e[number][8934],e[number][8935],e[number][8936],e[number][8937],e[number][8938],e[number][8939],e[number][8940],\
e[number][8941],e[number][8942],e[number][8943],e[number][8944],e[number][8945],e[number][8946],e[number][8947],e[number][8948],e[number][8949],e[number][8950],\
e[number][8951],e[number][8952],e[number][8953],e[number][8954],e[number][8955],e[number][8956],e[number][8957],e[number][8958],e[number][8959],e[number][8960],\
e[number][8961],e[number][8962],e[number][8963],e[number][8964],e[number][8965],e[number][8966],e[number][8967],e[number][8968],e[number][8969],e[number][8970],\
e[number][8971],e[number][8972],e[number][8973],e[number][8974],e[number][8975],e[number][8976],e[number][8977],e[number][8978],e[number][8979],e[number][8980],\
e[number][8981],e[number][8982],e[number][8983],e[number][8984],e[number][8985],e[number][8986],e[number][8987],e[number][8988],e[number][8989],e[number][8990],\
e[number][8991],e[number][8992],e[number][8993],e[number][8994],e[number][8995],e[number][8996],e[number][8997],e[number][8998],e[number][8999],e[number][9000],\
e[number][9001],e[number][9002],e[number][9003],e[number][9004],e[number][9005],e[number][9006],e[number][9007],e[number][9008],e[number][9009],e[number][9010],\
e[number][9011],e[number][9012],e[number][9013],e[number][9014],e[number][9015],e[number][9016],e[number][9017],e[number][9018],e[number][9019],e[number][9020],\
e[number][9021],e[number][9022],e[number][9023],e[number][9024],e[number][9025],e[number][9026],e[number][9027],e[number][9028],e[number][9029],e[number][9030],\
e[number][9031],e[number][9032],e[number][9033],e[number][9034],e[number][9035],e[number][9036],e[number][9037],e[number][9038],e[number][9039],e[number][9040],\
e[number][9041],e[number][9042],e[number][9043],e[number][9044],e[number][9045],e[number][9046],e[number][9047],e[number][9048],e[number][9049],e[number][9050],\
e[number][9051],e[number][9052],e[number][9053],e[number][9054],e[number][9055],e[number][9056],e[number][9057],e[number][9058],e[number][9059],e[number][9060],\
e[number][9061],e[number][9062],e[number][9063],e[number][9064],e[number][9065],e[number][9066],e[number][9067],e[number][9068],e[number][9069],e[number][9070],\
e[number][9071],e[number][9072],e[number][9073],e[number][9074],e[number][9075],e[number][9076],e[number][9077],e[number][9078],e[number][9079],e[number][9080],\
e[number][9081],e[number][9082],e[number][9083],e[number][9084],e[number][9085],e[number][9086],e[number][9087],e[number][9088],e[number][9089],e[number][9090],\
e[number][9091],e[number][9092],e[number][9093],e[number][9094],e[number][9095],e[number][9096],e[number][9097],e[number][9098],e[number][9099],e[number][9100],\
e[number][9101],e[number][9102],e[number][9103],e[number][9104],e[number][9105],e[number][9106],e[number][9107],e[number][9108],e[number][9109],e[number][9110],\
e[number][9111],e[number][9112],e[number][9113],e[number][9114],e[number][9115],e[number][9116],e[number][9117],e[number][9118],e[number][9119],e[number][9120],\
e[number][9121],e[number][9122],e[number][9123],e[number][9124],e[number][9125],e[number][9126],e[number][9127],e[number][9128],e[number][9129],e[number][9130],\
e[number][9131],e[number][9132],e[number][9133],e[number][9134],e[number][9135],e[number][9136],e[number][9137],e[number][9138],e[number][9139],e[number][9140],\
e[number][9141],e[number][9142],e[number][9143],e[number][9144],e[number][9145],e[number][9146],e[number][9147],e[number][9148],e[number][9149],e[number][9150],\
e[number][9151],e[number][9152],e[number][9153],e[number][9154],e[number][9155],e[number][9156],e[number][9157],e[number][9158],e[number][9159],e[number][9160],\
e[number][9161],e[number][9162],e[number][9163],e[number][9164],e[number][9165],e[number][9166],e[number][9167],e[number][9168],e[number][9169],e[number][9170],\
e[number][9171],e[number][9172],e[number][9173],e[number][9174],e[number][9175],e[number][9176],e[number][9177],e[number][9178],e[number][9179],e[number][9180],\
e[number][9181],e[number][9182],e[number][9183],e[number][9184],e[number][9185],e[number][9186],e[number][9187],e[number][9188],e[number][9189],e[number][9190],\
e[number][9191],e[number][9192],e[number][9193],e[number][9194],e[number][9195],e[number][9196],e[number][9197],e[number][9198],e[number][9199],e[number][9200],\
e[number][9201],e[number][9202],e[number][9203],e[number][9204],e[number][9205],e[number][9206],e[number][9207],e[number][9208],e[number][9209],e[number][9210],\
e[number][9211],e[number][9212],e[number][9213],e[number][9214],e[number][9215],e[number][9216],e[number][9217],e[number][9218],e[number][9219],e[number][9220],\
e[number][9221],e[number][9222],e[number][9223],e[number][9224],e[number][9225],e[number][9226],e[number][9227],e[number][9228],e[number][9229],e[number][9230],\
e[number][9231],e[number][9232],e[number][9233],e[number][9234],e[number][9235],e[number][9236],e[number][9237],e[number][9238],e[number][9239],e[number][9240],\
e[number][9241],e[number][9242],e[number][9243],e[number][9244],e[number][9245],e[number][9246],e[number][9247],e[number][9248],e[number][9249],e[number][9250],\
e[number][9251],e[number][9252],e[number][9253],e[number][9254],e[number][9255],e[number][9256],e[number][9257],e[number][9258],e[number][9259],e[number][9260],\
e[number][9261],e[number][9262],e[number][9263],e[number][9264],e[number][9265],e[number][9266],e[number][9267],e[number][9268],e[number][9269],e[number][9270],\
e[number][9271],e[number][9272],e[number][9273],e[number][9274],e[number][9275],e[number][9276],e[number][9277],e[number][9278],e[number][9279],e[number][9280],\
e[number][9281],e[number][9282],e[number][9283],e[number][9284],e[number][9285],e[number][9286],e[number][9287],e[number][9288],e[number][9289],e[number][9290],\
e[number][9291],e[number][9292],e[number][9293],e[number][9294],e[number][9295],e[number][9296],e[number][9297],e[number][9298],e[number][9299],e[number][9300],\
e[number][9301],e[number][9302],e[number][9303],e[number][9304],e[number][9305],e[number][9306],e[number][9307],e[number][9308],e[number][9309],e[number][9310],\
e[number][9311],e[number][9312],e[number][9313],e[number][9314],e[number][9315],e[number][9316],e[number][9317],e[number][9318],e[number][9319],e[number][9320],\
e[number][9321],e[number][9322],e[number][9323],e[number][9324],e[number][9325],e[number][9326],e[number][9327],e[number][9328],e[number][9329],e[number][9330],\
e[number][9331],e[number][9332],e[number][9333],e[number][9334],e[number][9335],e[number][9336],e[number][9337],e[number][9338],e[number][9339],e[number][9340],\
e[number][9341],e[number][9342],e[number][9343],e[number][9344],e[number][9345],e[number][9346],e[number][9347],e[number][9348],e[number][9349],e[number][9350],\
e[number][9351],e[number][9352],e[number][9353],e[number][9354],e[number][9355],e[number][9356],e[number][9357],e[number][9358],e[number][9359],e[number][9360],\
e[number][9361],e[number][9362],e[number][9363],e[number][9364],e[number][9365],e[number][9366],e[number][9367],e[number][9368],e[number][9369],e[number][9370],\
e[number][9371],e[number][9372],e[number][9373],e[number][9374],e[number][9375],e[number][9376],e[number][9377],e[number][9378],e[number][9379],e[number][9380],\
e[number][9381],e[number][9382],e[number][9383],e[number][9384],e[number][9385],e[number][9386],e[number][9387],e[number][9388],e[number][9389],e[number][9390],\
e[number][9391],e[number][9392],e[number][9393],e[number][9394],e[number][9395],e[number][9396],e[number][9397],e[number][9398],e[number][9399],e[number][9400],\
e[number][9401],e[number][9402],e[number][9403],e[number][9404],e[number][9405],e[number][9406],e[number][9407],e[number][9408],e[number][9409],e[number][9410],\
e[number][9411],e[number][9412],e[number][9413],e[number][9414],e[number][9415],e[number][9416],e[number][9417],e[number][9418],e[number][9419],e[number][9420],\
e[number][9421],e[number][9422],e[number][9423],e[number][9424],e[number][9425],e[number][9426],e[number][9427],e[number][9428],e[number][9429],e[number][9430],\
e[number][9431],e[number][9432],e[number][9433],e[number][9434],e[number][9435],e[number][9436],e[number][9437],e[number][9438],e[number][9439],e[number][9440],\
e[number][9441],e[number][9442],e[number][9443],e[number][9444],e[number][9445],e[number][9446],e[number][9447],e[number][9448],e[number][9449],e[number][9450],\
e[number][9451],e[number][9452],e[number][9453],e[number][9454],e[number][9455],e[number][9456],e[number][9457],e[number][9458],e[number][9459],e[number][9460],\
e[number][9461],e[number][9462],e[number][9463],e[number][9464],e[number][9465],e[number][9466],e[number][9467],e[number][9468],e[number][9469],e[number][9470],\
e[number][9471],e[number][9472],e[number][9473],e[number][9474],e[number][9475],e[number][9476],e[number][9477],e[number][9478],e[number][9479],e[number][9480],\
e[number][9481],e[number][9482],e[number][9483],e[number][9484],e[number][9485],e[number][9486],e[number][9487],e[number][9488],e[number][9489],e[number][9490],\
e[number][9491],e[number][9492],e[number][9493],e[number][9494],e[number][9495],e[number][9496],e[number][9497],e[number][9498],e[number][9499],e[number][9500],\
e[number][9501],e[number][9502],e[number][9503],e[number][9504],e[number][9505],e[number][9506],e[number][9507],e[number][9508],e[number][9509],e[number][9510],\
e[number][9511],e[number][9512],e[number][9513],e[number][9514],e[number][9515],e[number][9516],e[number][9517],e[number][9518],e[number][9519],e[number][9520],\
e[number][9521],e[number][9522],e[number][9523],e[number][9524],e[number][9525],e[number][9526],e[number][9527],e[number][9528],e[number][9529],e[number][9530],\
e[number][9531],e[number][9532],e[number][9533],e[number][9534],e[number][9535],e[number][9536],e[number][9537],e[number][9538],e[number][9539],e[number][9540],\
e[number][9541],e[number][9542],e[number][9543],e[number][9544],e[number][9545],e[number][9546],e[number][9547],e[number][9548],e[number][9549],e[number][9550],\
e[number][9551],e[number][9552],e[number][9553],e[number][9554],e[number][9555],e[number][9556],e[number][9557],e[number][9558],e[number][9559],e[number][9560],\
e[number][9561],e[number][9562],e[number][9563],e[number][9564],e[number][9565],e[number][9566],e[number][9567],e[number][9568],e[number][9569],e[number][9570],\
e[number][9571],e[number][9572],e[number][9573],e[number][9574],e[number][9575],e[number][9576],e[number][9577],e[number][9578],e[number][9579],e[number][9580],\
e[number][9581],e[number][9582],e[number][9583],e[number][9584],e[number][9585],e[number][9586],e[number][9587],e[number][9588],e[number][9589],e[number][9590],\
e[number][9591],e[number][9592],e[number][9593],e[number][9594],e[number][9595],e[number][9596],e[number][9597],e[number][9598],e[number][9599],e[number][9600],\
e[number][9601],e[number][9602],e[number][9603],e[number][9604],e[number][9605],e[number][9606],e[number][9607],e[number][9608],e[number][9609],e[number][9610],\
e[number][9611],e[number][9612],e[number][9613],e[number][9614],e[number][9615],e[number][9616],e[number][9617],e[number][9618],e[number][9619],e[number][9620],\
e[number][9621],e[number][9622],e[number][9623],e[number][9624],e[number][9625],e[number][9626],e[number][9627],e[number][9628],e[number][9629],e[number][9630],\
e[number][9631],e[number][9632],e[number][9633],e[number][9634],e[number][9635],e[number][9636],e[number][9637],e[number][9638],e[number][9639],e[number][9640],\
e[number][9641],e[number][9642],e[number][9643],e[number][9644],e[number][9645],e[number][9646],e[number][9647],e[number][9648],e[number][9649],e[number][9650],\
e[number][9651],e[number][9652],e[number][9653],e[number][9654],e[number][9655],e[number][9656],e[number][9657],e[number][9658],e[number][9659],e[number][9660],\
e[number][9661],e[number][9662],e[number][9663],e[number][9664],e[number][9665],e[number][9666],e[number][9667],e[number][9668],e[number][9669],e[number][9670],\
e[number][9671],e[number][9672],e[number][9673],e[number][9674],e[number][9675],e[number][9676],e[number][9677],e[number][9678],e[number][9679],e[number][9680],\
e[number][9681],e[number][9682],e[number][9683],e[number][9684],e[number][9685],e[number][9686],e[number][9687],e[number][9688],e[number][9689],e[number][9690],\
e[number][9691],e[number][9692],e[number][9693],e[number][9694],e[number][9695],e[number][9696],e[number][9697],e[number][9698],e[number][9699],e[number][9700],\
e[number][9701],e[number][9702],e[number][9703],e[number][9704],e[number][9705],e[number][9706],e[number][9707],e[number][9708],e[number][9709],e[number][9710],\
e[number][9711],e[number][9712],e[number][9713],e[number][9714],e[number][9715],e[number][9716],e[number][9717],e[number][9718],e[number][9719],e[number][9720],\
e[number][9721],e[number][9722],e[number][9723],e[number][9724],e[number][9725],e[number][9726],e[number][9727],e[number][9728],e[number][9729],e[number][9730],\
e[number][9731],e[number][9732],e[number][9733],e[number][9734],e[number][9735],e[number][9736],e[number][9737],e[number][9738],e[number][9739],e[number][9740],\
e[number][9741],e[number][9742],e[number][9743],e[number][9744],e[number][9745],e[number][9746],e[number][9747],e[number][9748],e[number][9749],e[number][9750],\
e[number][9751],e[number][9752],e[number][9753],e[number][9754],e[number][9755],e[number][9756],e[number][9757],e[number][9758],e[number][9759],e[number][9760],\
e[number][9761],e[number][9762],e[number][9763],e[number][9764],e[number][9765],e[number][9766],e[number][9767],e[number][9768],e[number][9769],e[number][9770],\
e[number][9771],e[number][9772],e[number][9773],e[number][9774],e[number][9775],e[number][9776],e[number][9777],e[number][9778],e[number][9779],e[number][9780],\
e[number][9781],e[number][9782],e[number][9783],e[number][9784],e[number][9785],e[number][9786],e[number][9787],e[number][9788],e[number][9789],e[number][9790],\
e[number][9791],e[number][9792],e[number][9793],e[number][9794],e[number][9795],e[number][9796],e[number][9797],e[number][9798],e[number][9799],e[number][9800],\
e[number][9801],e[number][9802],e[number][9803],e[number][9804],e[number][9805],e[number][9806],e[number][9807],e[number][9808],e[number][9809],e[number][9810],\
e[number][9811],e[number][9812],e[number][9813],e[number][9814],e[number][9815],e[number][9816],e[number][9817],e[number][9818],e[number][9819],e[number][9820],\
e[number][9821],e[number][9822],e[number][9823],e[number][9824],e[number][9825],e[number][9826],e[number][9827],e[number][9828],e[number][9829],e[number][9830],\
e[number][9831],e[number][9832],e[number][9833],e[number][9834],e[number][9835],e[number][9836],e[number][9837],e[number][9838],e[number][9839],e[number][9840],\
e[number][9841],e[number][9842],e[number][9843],e[number][9844],e[number][9845],e[number][9846],e[number][9847],e[number][9848],e[number][9849],e[number][9850],\
e[number][9851],e[number][9852],e[number][9853],e[number][9854],e[number][9855],e[number][9856],e[number][9857],e[number][9858],e[number][9859],e[number][9860],\
e[number][9861],e[number][9862],e[number][9863],e[number][9864],e[number][9865],e[number][9866],e[number][9867],e[number][9868],e[number][9869],e[number][9870],\
e[number][9871],e[number][9872],e[number][9873],e[number][9874],e[number][9875],e[number][9876],e[number][9877],e[number][9878],e[number][9879],e[number][9880],\
e[number][9881],e[number][9882],e[number][9883],e[number][9884],e[number][9885],e[number][9886],e[number][9887],e[number][9888],e[number][9889],e[number][9890],\
e[number][9891],e[number][9892],e[number][9893],e[number][9894],e[number][9895],e[number][9896],e[number][9897],e[number][9898],e[number][9899],e[number][9900],\
e[number][9901],e[number][9902],e[number][9903],e[number][9904],e[number][9905],e[number][9906],e[number][9907],e[number][9908],e[number][9909],e[number][9910],\
e[number][9911],e[number][9912],e[number][9913],e[number][9914],e[number][9915],e[number][9916],e[number][9917],e[number][9918],e[number][9919],e[number][9920],\
e[number][9921],e[number][9922],e[number][9923],e[number][9924],e[number][9925],e[number][9926],e[number][9927],e[number][9928],e[number][9929],e[number][9930],\
e[number][9931],e[number][9932],e[number][9933],e[number][9934],e[number][9935],e[number][9936],e[number][9937],e[number][9938],e[number][9939],e[number][9940],\
e[number][9941],e[number][9942],e[number][9943],e[number][9944],e[number][9945],e[number][9946],e[number][9947],e[number][9948],e[number][9949],e[number][9950],\
e[number][9951],e[number][9952],e[number][9953],e[number][9954],e[number][9955],e[number][9956],e[number][9957],e[number][9958],e[number][9959],e[number][9960],\
e[number][9961],e[number][9962],e[number][9963],e[number][9964],e[number][9965],e[number][9966],e[number][9967],e[number][9968],e[number][9969],e[number][9970],\
e[number][9971],e[number][9972],e[number][9973],e[number][9974],e[number][9975],e[number][9976],e[number][9977],e[number][9978],e[number][9979],e[number][9980],\
e[number][9981],e[number][9982],e[number][9983],e[number][9984],e[number][9985],e[number][9986],e[number][9987],e[number][9988],e[number][9989],e[number][9990],\
e[number][9991],e[number][9992],e[number][9993],e[number][9994],e[number][9995],e[number][9996],e[number][9997],e[number][9998],e[number][9999],another=r.split(",")


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        
        self.decoder1 = nn.Sequential(
            nn.Linear(10000, 8192),
            nn.Tanh(),
            nn.Linear(8192, 4096),
            nn.Tanh(),
            )
        self.conv1 = nn.Sequential(        # input shape (1, 128, 128)

            nn.Conv2d(

                in_channels=1,              # input height

                out_channels=50,            # n_filters

                kernel_size=3,              # filter size

                stride=1,                   # filter movement/step

                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1

            ),                              # output shape (50, 128, 128)

            nn.ReLU(),                      # activation
            
            #pooling 
            nn.AdaptiveMaxPool2d((64,64)),    # choose max value in 2x2 area, output shape (50, 128, 128)
        )
            
  
        self.block1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),
            nn.BatchNorm2d(50),                       
            nn.ReLU(),
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),
            nn.BatchNorm2d(50),                       
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),
            nn.BatchNorm2d(50),                       
            nn.ReLU(),
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),
            nn.BatchNorm2d(50),                       
            nn.ReLU(),
        )   
            
        self.block3 = nn.Sequential(         
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),
            nn.BatchNorm2d(50),                       
            nn.ReLU(),
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),
            nn.BatchNorm2d(50),                       
            nn.ReLU(),
        )   
        
        self.block4 = nn.Sequential(         
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),
            nn.BatchNorm2d(50),                       
            nn.ReLU(),
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),
            nn.Conv2d(
                in_channels=50,              
                out_channels=50,             
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),
            nn.BatchNorm2d(50),                       
            nn.ReLU(),
        )   

    ### END ###   (50,128,128)
        
        self.conv2 = nn.Sequential(         # input shape (50, 128, 128)

            nn.Conv2d(50, 1, 3, 1, 1),     # output shape (50, 128, 128)

            nn.ReLU(),                      # activation

        )

    def forward(self, x):
        #m=torch.zeros(100,64,64)
        #m=m.type(torch.cuda.FloatTensor)
        x = self.decoder1(x)
        x=x.reshape(BATCH_SIZE,64,64)
        x=x.unsqueeze(0)
        '''x=x.transpose(1,0)
        x= self.conv(x)
        x=x.transpose(1,0)
        x=x.reshape(BATCH_SIZE,4096)'''
        x = self.conv1(x)

        residual1 = x    #Save input as residual
        x = self.block1(x)
    
        x += residual1 #add input to output of block1
        residual2 = x  #save output of block1 as residual
        
        x = self.block2(x)
        
        x += residual2 #add output of block1 to output of block2
        residual3 = x
        
        x = self.block3(x)
        
        x += residual3 #add output of block2 to output of block3
        residual4 = x
        
        x = self.block4(x)
        x += residual4 #add output of block3 to output of block4
        
        decoded = self.conv2(x)
        return decoded


autoencoder = AutoEncoder()
autoencoder=autoencoder.cuda()
#autoencoder=torch.load("/home/why/unn1/unn/create.pt")
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()


p=list(range(10000))
inp=list(range(BATCH_SIZE))
inp2=torch.zeros(BATCH_SIZE,10000)

dataloader = DataLoader(ImageFolder('/home/why/unn1/unn/origin/', img_size=64),
                        batch_size=1, shuffle=False)




for epoch in range(EPOCH):
    for step, (img_paths, input_imgs) in enumerate(dataloader):
        #if(step==937): break

        for bit in range(BATCH_SIZE):
         for bon in range(10000):
          p[bon]=float(e[(step)*BATCH_SIZE+bit][bon])
         inp[bit]=p
        inp2=torch.tensor(inp, dtype=torch.float)
        #print(b_label) 
        #if(int(b_label[0])==9):
         #print(inp2.shape)
        #b_x=input_imgs.view(1,1,64*64)
        inp2=inp2.type(torch.cuda.FloatTensor)
        input_imgs=input_imgs.type(torch.cuda.FloatTensor)
         #b_y=b_y.type(torch.cuda.FloatTensor)
        decoded = autoencoder(inp2)

        loss = loss_func(decoded[0], input_imgs)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if step % 1 == 0:
             loss=loss.type(torch.FloatTensor)
             print('Epoch: ', epoch ,step, '| train loss: %.4f' % loss.data.numpy())      
             loss=loss.type(torch.cuda.FloatTensor)
        if step % 50 == 0:
             decoded=decoded.type(torch.FloatTensor)
             plt.imshow(decoded.data.numpy()[0][0],cmap='gray')
             plt.show()
             decoded=decoded.type(torch.cuda.FloatTensor)
    torch.save(autoencoder,"/home/why/unn1/unn/create.pt")
