from __future__ import division

from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import scipy

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
 

EPOCH = 100
BATCH_SIZE = 1
LR = 0.001         # learning rate

f=open("./last_data.txt","r")
e=list(range(4500))
for number in range(4500):
   e[number]=list(range(2048))
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
   e[number][2040],e[number][2041],e[number][2042],e[number][2043],e[number][2044],e[number][2045],e[number][2046],e[number][2047],another=r.split(",")        #64*64*0.5= 2048 SR=0.5 as examples


class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()


        self.decoder1 = nn.Sequential(
            nn.Linear(2048, 8192),
            nn.Tanh(),
            nn.Linear(8192, 4096),
            nn.Tanh(),
            )

        self.conv1 = nn.Sequential(        # input shape (1, 64, 64)

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
        
        output = self.conv2(x)
  
        #output = self.out(x)   #(1,64,64)

        return output    # return x for visualization



autoencoder = CNN()
autoencoder=autoencoder.cuda()
autoencoder.train()
autoencoder=torch.load("/home/divinezeng/unn/netD.pt")
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()









dataloader1 = DataLoader(ImageFolder('/home/why/unn/origin/', img_size=64),
                        batch_size=1, shuffle=False)

dataloader2 = DataLoader(ImageFolder('/home/why/unn/origin/', img_size=64),
                        batch_size=1, shuffle=False)



for epoch in range(EPOCH):
    torch.save(autoencoder,"/home/divinezeng/unn/netD.pt")
    for a,b in zip(enumerate(dataloader1),enumerate(dataloader2)):

        in_img=a[1][1]
        f=open("/home/divinezeng/unn/3d/"+str(a[0]+1)+".txt","r")
        e=np.eye(64)
        for number in range(64):
           r=f.readline()
           e[number][0],e[number][1],e[number][2],e[number][3],e[number][4],e[number][5],e[number][6],e[number][7],e[number][8],e[number][9],e[number][10],e[number][11],e[number][12],e[number][13],e[number][14],e[number][15],e[number][16],e[number][17],e[number][18],e[number][19],e[number][20],e[number][21],e[number][22],e[number][23],e[number][24],e[number][25],e[number][26],e[number][27],e[number][28],e[number][29],e[number][30],e[number][31],e[number][32],e[number][33],e[number][34],e[number][35],e[number][36],e[number][37],e[number][38],e[number][39],e[number][40],e[number][41],e[number][42],e[number][43],e[number][44],e[number][45],e[number][46],e[number][47],e[number][48],e[number][49],e[number][50],e[number][51],e[number][52],e[number][53],e[number][54],e[number][55],e[number][56],e[number][57],e[number][58],e[number][59],e[number][60],e[number][61],e[number][62],e[number][63]=r.split(",")
           '''for x in range(64):
              e[number][x]=e[number][x]*10'''

        e=torch.Tensor(e)
        target=e
        in_img=in_img.unsqueeze(0)
        in_img=in_img.type(torch.cuda.FloatTensor)
        target=target.type(torch.cuda.FloatTensor)
        decoded = autoencoder(in_img)
        loss = loss_func(decoded[0], target)# mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if a[0] % 1 == 0:
             loss=loss.type(torch.FloatTensor)
             decoded=decoded.type(torch.FloatTensor)
             print('Epoch: ', epoch ,a[0], '| train loss: %.4f' % loss.data.numpy())
             #scipy.misc.imsave("/home/nvidia/Desktop/unn/f/re"+str(a[0])+".jpg",decoded.data.numpy()[0][0])
             loss=loss.type(torch.cuda.FloatTensor)
             decoded=decoded.type(torch.cuda.FloatTensor)

        if a[0] % 100 == 0:
             decoded=decoded.type(torch.FloatTensor)
             target=target.type(torch.FloatTensor)
             #test out####################################################33
             fig = plt.figure()
             ax = fig.gca(projection='3d')
             buff=decoded[0].data.numpy()[0]
             X = np.arange(-16, 16, 0.5)
             Y = np.arange(-16, 16, 0.5)
             X, Y = np.meshgrid(X, Y)
             Z = X+Y
             for x in range(64):
                for y in range(64):
                  Z[x][y]=buff[x][y]
             surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
             ax.set_zlim(0, 40)
             ax.zaxis.set_major_locator(LinearLocator(10))
             ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
             fig.colorbar(surf, shrink=0.5, aspect=5)
             plt.show()

             target=target.type(torch.cuda.FloatTensor)
             decoded=decoded.type(torch.cuda.FloatTensor)
             ####################################################################
             #yuan tu ##########################################################
             '''fig = plt.figure()
             ax = fig.gca(projection='3d')
             X = np.arange(-16, 16, 0.5)
             Y = np.arange(-16, 16, 0.5)
             X, Y = np.meshgrid(X, Y)
             Z = X+Y
             for x in range(64):
                for y in range(64):
                  Z[x][y]=e[x][y]
             surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
             ax.set_zlim(0, 40)
             ax.zaxis.set_major_locator(LinearLocator(10))
             ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
             fig.colorbar(surf, shrink=0.5, aspect=5)
             plt.show()'''
             #####################################################################
             '''cm=plt.cm.get_cmap('bwr')
             xy=range(0)
             z=xy
             sc=plt.scatter(xy,xy,c=z,vmin=-1.5,vmax=1.5,s=35,cmap=cm)
             plt.colorbar(sc)
             test=(target-decoded).data.numpy()[0][0]
             plt.imshow(test,cmap=cm)
             plt.show()'''
             target=target.type(torch.cuda.FloatTensor)
             decoded=decoded.type(torch.cuda.FloatTensor)
