#mprof run ASC_mpl.py
#mprof plot

import time

import requests
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
import pandas
import numpy
from skyfield import almanac, almanac_east_asia
from skyfield.api import load, Topos, PlanetaryConstants
from skyfield.framelib import ecliptic_frame
from skyfield.almanac import find_discrete, risings_and_settings
from skyfield.magnitudelib import planetary_magnitude
from pytz import timezone, common_timezones
from datetime import date, datetime, timedelta
import cartopy.crs as ccrs
import pathlib
from PIL import Image
import itertools
import sys
from sys import platform
import os
import gc
import feedparser
import re
import objgraph

##################
#memory leak
debug_mode = 0
##################

######################
# initial parameters #
######################

count=0
T0 = time.time()

#####################################
# ephem setting
tz          = timezone('Asia/Hong_Kong')

ephem       = load('de421.bsp') #1900-2050 only
#ephem      = load('de422.bsp') #-3000-3000 only
#ephem      = load('de430t.bsp') #1550-2650 only
sun         = ephem['sun']
earthmoon   = ephem['earth_barycenter']
earth       = ephem['earth']
moon        = ephem['moon']
#####################################
# moon geometry
pc = PlanetaryConstants()
pc.read_text(load('moon_080317.tf'))
pc.read_text(load('pck00008.tpc'))
pc.read_binary(load('moon_pa_de421_1900-2050.bpc'))
frame = pc.build_frame_named('MOON_ME_DE421')
#####################################

#####################################
# location information
#HKO
Trig_0      = (earth + Topos(str(22+18/60+7.3/3600)+' N', str(114+10/60+27.6/3600)+' E'),\
               22+18/60+7.3/3600,114+10/60+27.6/3600,'22:18:07.3','N','114:10:27.6','E')

#Hokoon
hokoon      = (earth + Topos(str(22+23/60+1/3600)+' N', str(114+6/60+29/3600)+' E'),\
               22+23/60+1/3600,114+6/60+29/3600,'22:23:01','N','114:06:29','E')

OBS         = hokoon #<= set your observatory

ts          = load.timescale()
date_UTC    = ts.utc(ts.now().utc_datetime().replace(second=0,microsecond=0))
date_local  = date_UTC.astimezone(tz)
#####################################
# plot parameters
image_size = 1.6
fig = plt.figure(figsize=(image_size*4,image_size*4), facecolor='black')
fig.subplots_adjust(0,0,1,1,0,0)

ax1 = plt.subplot()
ax1.set_facecolor('black')
ax1.set_aspect('equal', anchor='NE')

matplotlib.rcParams['savefig.facecolor'] = (0,0,0)

if platform == 'win32':
    DjV_S_6     = font_manager.FontProperties(fname = 'C:/WINDOWS/Fonts/DEJAVUSANS.TTF', size=6)
    DjV_S_8     = font_manager.FontProperties(fname = 'C:/WINDOWS/Fonts/DEJAVUSANS.TTF', size=8)
    DjV_S_9     = font_manager.FontProperties(fname = 'C:/WINDOWS/Fonts/DEJAVUSANS.TTF', size=9)
    DjV_S_10    = font_manager.FontProperties(fname = 'C:/WINDOWS/Fonts/DEJAVUSANS.TTF', size=10)
    DjV_S_12    = font_manager.FontProperties(fname = 'C:/WINDOWS/Fonts/DEJAVUSANS.TTF', size=12)
    emoji_20    = font_manager.FontProperties(fname = 'C:/WINDOWS/Fonts/YUGOTHB.TTC', size=20)
elif platform == 'darwin':
    DjV_S_6     = font_manager.FontProperties(fname = '/Library/Fonts/DEJAVUSANS.TTF', size=6)
    DjV_S_8     = font_manager.FontProperties(fname = '/Library/Fonts/DEJAVUSANS.TTF', size=8)
    DjV_S_9     = font_manager.FontProperties(fname = '/Library/Fonts/DEJAVUSANS.TTF', size=9)
    DjV_S_10    = font_manager.FontProperties(fname = '/Library/Fonts/DEJAVUSANS.TTF', size=10)
    DjV_S_12    = font_manager.FontProperties(fname = '/Library/Fonts/DEJAVUSANS.TTF', size=12)
    emoji_20    = font_manager.FontProperties(fname = '/Library/Fonts/YUGOTHB.TTC', size=20)
elif platform == 'linux':
    DjV_S_6     = font_manager.FontProperties(fname = '/usr/local/share/fonts/DejaVuSans.ttf', size=6)
    DjV_S_8     = font_manager.FontProperties(fname = '/usr/local/share/fonts/DejaVuSans.ttf', size=8)
    DjV_S_9     = font_manager.FontProperties(fname = '/usr/local/share/fonts/DejaVuSans.ttf', size=9)
    DjV_S_10    = font_manager.FontProperties(fname = '/usr/local/share/fonts/DejaVuSans.ttf', size=10)
    DjV_S_12    = font_manager.FontProperties(fname = '/usr/local/share/fonts/DejaVuSans.ttf', size=12)
    emoji_20    = font_manager.FontProperties(fname = '/usr/local/share/fonts/YuGothB.ttc', size=20)

# log
def timelog(log):
    print(str(datetime.now().time().replace(microsecond=0))+'> '+log)
    
# LROC WAC basemap Shapefile
Mare    = numpy.zeros(shape=(267482,5)) 
Mare    = pandas.read_csv(pathlib.Path.cwd().joinpath('ASC','moon_mare.csv'))
Crater  = numpy.zeros(shape=(182111,5))
Crater  = pandas.read_csv(pathlib.Path.cwd().joinpath('ASC','moon_crater.csv'))
    
####################
# define functions #
####################

def plot_solar():
    global plot_alpha, sun_vector, moon_vector, solar_obj, solar_color, moon_chi

    sun_vector      = OBS[0].at(date_UTC).observe(sun).apparent()
    moon_vector     = OBS[0].at(date_UTC).observe(moon).apparent()
    
    # position angle of the Moon's bright limb from North point of the disc of the Moon to East
    moon_chi = numpy.degrees(numpy.arctan2(numpy.cos(sun_vector.radec()[1].radians)*numpy.sin(sun_vector.radec()[0].radians-moon_vector.radec()[0].radians),
                                           numpy.sin(sun_vector.radec()[1].radians)*numpy.cos(moon_vector.radec()[1].radians)-
                                           numpy.cos(sun_vector.radec()[1].radians)*numpy.sin(moon_vector.radec()[1].radians)*numpy.cos(sun_vector.radec()[0].radians-moon_vector.radec()[0].radians))) % 360

    # alpha
    if sun_vector.altaz()[0].degrees >= 0:
        plot_alpha = 0.2
    else:
        plot_alpha = 0.1
    
    timelog('plotting solar system objects')
           
    solar_color = ['#FFCC33','#DAD9D7']
   
def moon_phase(): #ax1
    global ax1_moon

    timelog('drawing Moon')
    
    ax1.set_xlim((-90,90))
    ax1.set_ylim((-90,90))
    
    # Moon phase
    ax1.annotate('Moon Phase',(0,85),xycoords=('data'),ha='center',va='top',fontproperties=DjV_S_12,color='white')

    moon_size = numpy.degrees(3474.2/moon_vector.radec()[2].km)*3600

    M_d  = moon_size/2100*110
    ph_r = 50*moon_size/2100 # line inner radius
    ph_R = 60*moon_size/2100 # line outer radius
    ph_l = 68*moon_size/2100 # text radius
    
    # illuminated percentage
    Moon_percent = almanac.fraction_illuminated(ephem, 'moon', date_UTC)
    
    # rotate for position angle of the Moon's bright limb
    rot_pa_limb_moon = moon_chi-90

    # rotation angle from zenith to equatorial north clockwise
    sidereal_time = date_UTC.gmst+OBS[2]/15
    Moon_parallactic_angle = numpy.arctan2(numpy.sin(numpy.radians(sidereal_time*15)-moon_vector.radec()[0].radians),
                                           numpy.tan(OBS[1])*numpy.cos(moon_vector.radec()[1].radians)-numpy.sin(moon_vector.radec()[1].radians)*numpy.cos(numpy.radians(sidereal_time*15)-moon_vector.radec()[0].radians))

    ##brightLimbAngle = (moon_chi - numpy.degrees(Moon_parallactic_angle))%360
    
    moondisc0 = patches.Circle((0,0), M_d/2, color='#F0F0F0')
    if Moon_percent == 0:
        moondisc1 = patches.Circle((0,0), M_d/2, color='#94908D')
        moondisc2 = patches.Wedge((0,0), 0, 0, 0, color='#94908D') #dummy
        moondisc3 = patches.Ellipse((0,0), 0, 0, 0, color='#94908D') #dummy
    elif 0 < Moon_percent < 0.5:
        moondisc1 = patches.Circle((0,0), M_d/2, color='#F0F0F0')
        moondisc2 = patches.Wedge((0,0), M_d/2, 270+rot_pa_limb_moon-numpy.degrees(Moon_parallactic_angle), 90+rot_pa_limb_moon-numpy.degrees(Moon_parallactic_angle), color='#94908D')
        moondisc3 = patches.Ellipse((0,0), M_d*(1-Moon_percent/0.5), M_d, angle=rot_pa_limb_moon-numpy.degrees(Moon_parallactic_angle), color='#94908D')
    elif Moon_percent == 0.5:
        moondisc1 = patches.Circle((0,0), M_d/2, color='#94908D')
        moondisc2 = patches.Wedge((0,0), M_d/2, 90+rot_pa_limb_moon-numpy.degrees(Moon_parallactic_angle), 270+rot_pa_limb_moon-numpy.degrees(Moon_parallactic_angle), color='#F0F0F0')
        moondisc3 = patches.Ellipse((0,0), 0, 0, 0, color='#F0F0F0') #dummy
    elif 0.5 < Moon_percent < 1:
        moondisc1 = patches.Circle((0,0), M_d/2, color='#94908D')
        moondisc2 = patches.Wedge((0,0), M_d/2, 90+rot_pa_limb_moon-numpy.degrees(Moon_parallactic_angle), 270+rot_pa_limb_moon-numpy.degrees(Moon_parallactic_angle), color='#F0F0F0')
        moondisc3 = patches.Ellipse((0,0), M_d*(1-Moon_percent/0.5), M_d, angle=rot_pa_limb_moon-numpy.degrees(Moon_parallactic_angle), color='#F0F0F0')
    elif Moon_percent == 1:
        moondisc1 = patches.Circle((0,0), M_d/2, color='#F0F0F0')
        moondisc2 = patches.Wedge((0,0), 0, 0, 0,color='#F0F0F0') #dummy
        moondisc3 = patches.Ellipse((0,0), 0, 0, 0, color='#F0F0F0') #dummy
    
    ax1.add_patch(moondisc0)
    ax1.add_patch(moondisc1)
    ax1.add_patch(moondisc2)
    ax1.add_patch(moondisc3)
    
    #libration
    Mlat, Mlon, distance = (earth - moon).at(date_UTC).frame_latlon(frame)

    T               = (date_UTC.tdb-2451545)/36525 # should use Julian Emphemeris Date instead
    asc_node        = 125.04452-1934.136261*T\
                      +0.0020708*T*T\
                      +T*T*T/450000 # longitude of ascending node of Moon mean orbit
    L_s             = 280.4665+36000.7698*T # mean longitude of Sun
    L_m             = 218.3165+481267.8813*T # mean longitude of Moon
    nu_lon          = -17.2/3600*numpy.sin(numpy.radians(asc_node))\
                      -1.32/3600*numpy.sin(numpy.radians(2*L_s))\
                      -0.23/3600*numpy.sin(numpy.radians(2*L_m))\
                      +0.21/3600*numpy.sin(numpy.radians(2*asc_node)) # nutation in longitude
    Inc             = 1.54242 # inclination of mean lunar equator to ecliptic
    M_s             = 357.5291092\
                      +35999.0502909*T\
                      -0.0001536*T*T\
                      +T*T*T/24490000 # Sun mean anomaly
    M_m             = 134.9634114\
                      +477198.8676313*T\
                      +0.008997*T*T\
                      +T*T*T/69699\
                      -T*T*T*T/14712000 # Moon mean anomaly
    D_m             = 297.8502042\
                      +445267.1115168*T\
                      -0.00163*T*T\
                      +T*T*T/545868\
                      -T*T*T*T/113065000 # mean elongation of Moon
    F_m             = 93.2720993\
                      +483202.0175273*T\
                      -0.0034029*T*T\
                      -T*T*T/3526000\
                      +T*T*T*T/863310000 # Moon argument of latitude
    rho             = -0.02752*numpy.cos(numpy.radians(M_m))\
                      -0.02245*numpy.sin(numpy.radians(F_m))\
                      +0.00684*numpy.cos(numpy.radians(M_m-2*F_m))\
                      -0.00293*numpy.cos(numpy.radians(2*F_m))\
                      -0.00085*numpy.cos(numpy.radians(2*F_m-2*D_m))\
                      -0.00054*numpy.cos(numpy.radians(M_m-2*D_m))\
                      -0.0002*numpy.sin(numpy.radians(M_m+F_m))\
                      -0.0002*numpy.cos(numpy.radians(M_m+2*F_m))\
                      -0.0002*numpy.cos(numpy.radians(M_m-F_m))\
                      +0.00014*numpy.cos(numpy.radians(M_m+2*F_m-2*D_m))
    sigma           = -0.02816*numpy.sin(numpy.radians(M_m))\
                      +0.02244*numpy.cos(numpy.radians(F_m))\
                      -0.00682*numpy.sin(numpy.radians(M_m-2*F_m))\
                      -0.00279*numpy.sin(numpy.radians(2*F_m))\
                      -0.00083*numpy.sin(numpy.radians(2*F_m-2*D_m))\
                      +0.00069*numpy.sin(numpy.radians(M_m-2*D_m))\
                      +0.0004*numpy.cos(numpy.radians(M_m+F_m))\
                      -0.00025*numpy.sin(numpy.radians(2*M_m))\
                      -0.00023*numpy.sin(numpy.radians(M_m+2*F_m))\
                      +0.0002*numpy.cos(numpy.radians(M_m-F_m))\
                      +0.00019*numpy.sin(numpy.radians(M_m-F_m))\
                      +0.00013*numpy.sin(numpy.radians(M_m+2*F_m-2*D_m))\
                      -0.0001*numpy.cos(numpy.radians(M_m-3*F_m))
    V_m             = asc_node + nu_lon + sigma/numpy.sin(numpy.radians(Inc))
    epsilion        = 23.4355636928 #(IAU 2000B nutation series)
    X_m             = numpy.sin(numpy.radians(Inc)+rho)*numpy.sin(numpy.radians(V_m))
    Y_m             = numpy.sin(numpy.radians(Inc)+rho)*numpy.cos(numpy.radians(V_m))*numpy.cos(numpy.radians(epsilion))\
                      -numpy.cos(numpy.radians(Inc)+rho)*numpy.sin(numpy.radians(epsilion))
    omega           = numpy.arctan2(X_m,Y_m)

    PA_axis_moon_N  = numpy.arcsin(numpy.sqrt(X_m*X_m+Y_m*Y_m)*numpy.cos(moon_vector.radec()[0].radians-omega)/numpy.cos(Mlon.radians))
            
    PA_axis_moon_z  = Moon_parallactic_angle-PA_axis_moon_N # clockwise, radians
    
    moon_rot = -numpy.degrees(PA_axis_moon_z) # anti-clockwise
    
    # Mare in Orthographic projection with rotation shown on ax1_moon
    lon0 = Mlon.degrees
    lat0 = Mlat.degrees
    
##    if count != 1:
##        ax1_moon.remove()
        
    fig0 = plt.figure(0)
    ax_moon_img = plt.axes(projection=ccrs.Orthographic(central_longitude=lon0,central_latitude=lat0))
    ax_moon_img.set_facecolor('none')
    ax_moon_img.axis('off')
    ax_moon_img.imshow(plt.imread('moonmap.png'), extent=(-180,180,-90,90), transform=ccrs.PlateCarree())
    fig0.tight_layout(pad=0)
    fig0.savefig('moon_proj.png', bbox_inches='tight', transparent=True)
    plt.close(fig0)
    
    ax1_moon = fig.add_axes([ax1.get_position().x0,ax1.get_position().y0,
                              ax1.get_position().width,ax1.get_position().height], projection=ccrs.Orthographic(central_longitude=lon0,central_latitude=lat0))
    ax1_moon.set_facecolor('none')
    ax1_moon.set_xlim((-90,90))
    ax1_moon.set_ylim((-90,90))
    ax1_moon.axis('off')

    moon_proj = Image.open(pathlib.Path.cwd().joinpath('moon_proj.png')).rotate(moon_rot)
    mscale = 1.045
    ax1_moon.imshow(moon_proj, extent=[-M_d/2*mscale,M_d/2*mscale,-M_d/2*mscale,M_d/2*mscale])

    # eq. coord.
    if moon_vector.altaz()[0].degrees > 0:
        ax1.annotate('N',(ph_l*numpy.sin(Moon_parallactic_angle),ph_l*numpy.cos(Moon_parallactic_angle)),\
                     xycoords=('data'),rotation=-numpy.degrees(Moon_parallactic_angle),ha='center',va='center',color='red')
        ax1.plot([ph_r*numpy.sin(Moon_parallactic_angle),ph_R*numpy.sin(Moon_parallactic_angle)],\
                 [ph_r*numpy.cos(Moon_parallactic_angle),ph_R*numpy.cos(Moon_parallactic_angle)],color='red')
        
        ax1.annotate('E',(ph_l*numpy.sin(Moon_parallactic_angle+3*numpy.pi/2),ph_l*numpy.cos(Moon_parallactic_angle+3*numpy.pi/2)),\
                     xycoords=('data'),rotation=-numpy.degrees(Moon_parallactic_angle),ha='center',va='center',color='red')
        ax1.plot([ph_r*numpy.sin(Moon_parallactic_angle+3*numpy.pi/2),ph_R*numpy.sin(Moon_parallactic_angle+3*numpy.pi/2)],\
                 [ph_r*numpy.cos(Moon_parallactic_angle+3*numpy.pi/2),ph_R*numpy.cos(Moon_parallactic_angle+3*numpy.pi/2)],color='red')
        
        ax1.annotate('S',(ph_l*numpy.sin(Moon_parallactic_angle+numpy.pi),ph_l*numpy.cos(Moon_parallactic_angle+numpy.pi)),\
                     xycoords=('data'),rotation=-numpy.degrees(Moon_parallactic_angle),ha='center',va='center',color='red')
        ax1.plot([ph_r*numpy.sin(Moon_parallactic_angle+numpy.pi),ph_R*numpy.sin(Moon_parallactic_angle+numpy.pi)],\
                 [ph_r*numpy.cos(Moon_parallactic_angle+numpy.pi),ph_R*numpy.cos(Moon_parallactic_angle+numpy.pi)],color='red')
        
        ax1.annotate('W',(ph_l*numpy.sin(Moon_parallactic_angle+numpy.pi/2),ph_l*numpy.cos(Moon_parallactic_angle+numpy.pi/2)),\
                     xycoords=('data'),rotation=-numpy.degrees(Moon_parallactic_angle),ha='center',va='center',color='red')
        ax1.plot([ph_r*numpy.sin(Moon_parallactic_angle+numpy.pi/2),ph_R*numpy.sin(Moon_parallactic_angle+numpy.pi/2)],\
                 [ph_r*numpy.cos(Moon_parallactic_angle+numpy.pi/2),ph_R*numpy.cos(Moon_parallactic_angle+numpy.pi/2)],color='red')

    ax1.annotate('eq \ncoord.',(-90,-70),xycoords=('data'),ha='left',va='bottom',fontproperties=DjV_S_9,color='red')

    # selenographic
    ax1.annotate('seleno-\ngraphic',(90,-70),xycoords=('data'),ha='right',va='bottom',fontproperties=DjV_S_9,color='cyan')

    ax1.annotate('N',(ph_l*numpy.sin(PA_axis_moon_z),ph_l*numpy.cos(PA_axis_moon_z)),\
                 xycoords=('data'),rotation=-numpy.degrees(PA_axis_moon_z),ha='center',va='center',color='cyan')
    ax1.plot([ph_r*numpy.sin(PA_axis_moon_z),ph_R*numpy.sin(PA_axis_moon_z)],\
             [ph_r*numpy.cos(PA_axis_moon_z),ph_R*numpy.cos(PA_axis_moon_z)],color='cyan')
    
    ax1.annotate('E',(ph_l*numpy.sin(PA_axis_moon_z+numpy.pi/2),ph_l*numpy.cos(PA_axis_moon_z+numpy.pi/2)),\
                 xycoords=('data'),rotation=-numpy.degrees(PA_axis_moon_z),ha='center',va='center',color='cyan')
    ax1.plot([ph_r*numpy.sin(PA_axis_moon_z+numpy.pi/2),ph_R*numpy.sin(PA_axis_moon_z+numpy.pi/2)],\
             [ph_r*numpy.cos(PA_axis_moon_z+numpy.pi/2),ph_R*numpy.cos(PA_axis_moon_z+numpy.pi/2)],color='cyan')
    
    ax1.annotate('S',(ph_l*numpy.sin(PA_axis_moon_z+numpy.pi),ph_l*numpy.cos(PA_axis_moon_z+numpy.pi)),\
                 xycoords=('data'),rotation=-numpy.degrees(PA_axis_moon_z),ha='center',va='center',color='cyan')
    ax1.plot([ph_r*numpy.sin(PA_axis_moon_z+numpy.pi),ph_R*numpy.sin(PA_axis_moon_z+numpy.pi)],\
             [ph_r*numpy.cos(PA_axis_moon_z+numpy.pi),ph_R*numpy.cos(PA_axis_moon_z+numpy.pi)],color='cyan')
    
    ax1.annotate('W',(ph_l*numpy.sin(PA_axis_moon_z+3*numpy.pi/2),ph_l*numpy.cos(PA_axis_moon_z+3*numpy.pi/2)),\
                 xycoords=('data'),rotation=-numpy.degrees(PA_axis_moon_z),ha='center',va='center',color='cyan')
    ax1.plot([ph_r*numpy.sin(PA_axis_moon_z+3*numpy.pi/2),ph_R*numpy.sin(PA_axis_moon_z+3*numpy.pi/2)],\
             [ph_r*numpy.cos(PA_axis_moon_z+3*numpy.pi/2),ph_R*numpy.cos(PA_axis_moon_z+3*numpy.pi/2)],color='cyan')
    
    # zenith
    if moon_vector.altaz()[0].degrees > 0:
        ax1.arrow(0,M_d/2,0,10,color='green',head_width=5, head_length=5)
        ax1.annotate(str(round(moon_size/60,1))+"'",(90,70),xycoords=('data'),ha='right',va='top',fontproperties=DjV_S_9,color='orange')
    else:
        ax1.annotate('below horizon',(90,70),xycoords=('data'),ha='right',va='top',fontproperties=DjV_S_9,color='orange')
    ax1.annotate('zenith',(-90,70),xycoords=('data'),ha='left',va='top',fontproperties=DjV_S_9,color='green')
    
    phase_moon = 'illuminated '+str(round(Moon_percent*100,2))+'%' # projected 2D apparent area
    if Moon_percent >= 0:
        ax1.annotate(phase_moon,(0,-85),xycoords=('data'),ha='center',va='bottom',fontproperties=DjV_S_10,color='#F0F0F0')
    else:
        ax1.annotate(phase_moon,(0,-85),xycoords=('data'),ha='center',va='bottom',fontproperties=DjV_S_10,color='#94908D')


plot_solar()
moon_phase()

# plot
fig.canvas.draw() 
fig.canvas.flush_events()
#plt.savefig('moon.eps')

plt.show()

