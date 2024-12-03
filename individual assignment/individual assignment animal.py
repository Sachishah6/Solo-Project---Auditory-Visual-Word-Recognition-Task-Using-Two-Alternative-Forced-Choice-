#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.5),
    on November 19, 2024, at 16:02
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.5'
expName = 'individual assignment animal'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1280, 720]
_loggingLevel = logging.getLevel('warning')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\HP\\Downloads\\individual assignment\\stimuli\\individual assignment animal.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = True
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_instruct') is None:
        # initialise key_instruct
        key_instruct = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_instruct',
        )
    # create speaker 'trail_sound'
    deviceManager.addDevice(
        deviceName='trail_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "welcome" ---
    text_norm = visual.TextStim(win=win, name='text_norm',
        text='Press the yellow star and do as intructed ',
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct = keyboard.Keyboard(deviceName='key_instruct')
    # Run 'Begin Experiment' code from text_align
    # Code components should usually appear at the top
    # of the routine. This one has to appear after the
    # text component it refers to.
    text_norm.alignText= 'left'
    
    # --- Initialize components for Routine "start" ---
    start_Lrectangle = visual.Rect(
        win=win, name='start_Lrectangle',
        width=(0.5, 0.5)[0], height=(0.5, 0.5)[1],
        ori=0.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    start_Rrecntangle = visual.Rect(
        win=win, name='start_Rrecntangle',
        width=(0.5, 0.5)[0], height=(0.5, 0.5)[1],
        ori=0.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    start_L_image = visual.ImageStim(
        win=win,
        name='start_L_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.4, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    start_R_image = visual.ImageStim(
        win=win,
        name='start_R_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0.4, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    star = visual.ShapeStim(
        win=win, name='star', vertices='star7',
        size=(0.15, 0.15),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='yellow', fillColor='yellow',
        opacity=None, depth=-4.0, interpolate=True)
    star_mouse = event.Mouse(win=win)
    x, y = [None, None]
    star_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "trial" ---
    right_rectanle = visual.Rect(
        win=win, name='right_rectanle',
        width=(0.5, 0.5)[0], height=(0.5, 0.5)[1],
        ori=0.0, pos=(0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    left_rectangle = visual.Rect(
        win=win, name='left_rectangle',
        width=(0.5, 0.5)[0], height=(0.5, 0.5)[1],
        ori=0.0, pos=(-0.4, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    left_image = visual.ImageStim(
        win=win,
        name='left_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.4, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    right_image = visual.ImageStim(
        win=win,
        name='right_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0.4, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    trail_sound = sound.Sound(
        'A', 
        secs=1.59, 
        stereo=True, 
        hamming=True, 
        speaker='trail_sound',    name='trail_sound'
    )
    trail_sound.setVolume(1.0)
    trial_mouse = event.Mouse(win=win)
    x, y = [None, None]
    trial_mouse.mouseClock = core.Clock()
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "welcome" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('welcome.started', globalClock.getTime(format='float'))
    # create starting attributes for key_instruct
    key_instruct.keys = []
    key_instruct.rt = []
    _key_instruct_allKeys = []
    # keep track of which components have finished
    welcomeComponents = [text_norm, key_instruct]
    for thisComponent in welcomeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_norm* updates
        
        # if text_norm is starting this frame...
        if text_norm.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_norm.frameNStart = frameN  # exact frame index
            text_norm.tStart = t  # local t and not account for scr refresh
            text_norm.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_norm, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_norm.status = STARTED
            text_norm.setAutoDraw(True)
        
        # if text_norm is active this frame...
        if text_norm.status == STARTED:
            # update params
            pass
        
        # *key_instruct* updates
        waitOnFlip = False
        
        # if key_instruct is starting this frame...
        if key_instruct.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instruct.frameNStart = frameN  # exact frame index
            key_instruct.tStart = t  # local t and not account for scr refresh
            key_instruct.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instruct.started')
            # update status
            key_instruct.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instruct_allKeys.extend(theseKeys)
            if len(_key_instruct_allKeys):
                key_instruct.keys = _key_instruct_allKeys[0].name  # just the first key pressed
                key_instruct.rt = _key_instruct_allKeys[0].rt
                key_instruct.duration = _key_instruct_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcomeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcomeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('welcome.stopped', globalClock.getTime(format='float'))
    # check responses
    if key_instruct.keys in ['', [], None]:  # No response was made
        key_instruct.keys = None
    thisExp.addData('key_instruct.keys',key_instruct.keys)
    if key_instruct.keys != None:  # we had a response
        thisExp.addData('key_instruct.rt', key_instruct.rt)
        thisExp.addData('key_instruct.duration', key_instruct.duration)
    thisExp.nextEntry()
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('C:/Users/HP/Downloads/trials.csv'),
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "start" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('start.started', globalClock.getTime(format='float'))
        start_L_image.setImage(l_image)
        start_R_image.setImage(r_image)
        # setup some python lists for storing info about the star_mouse
        star_mouse.x = []
        star_mouse.y = []
        star_mouse.leftButton = []
        star_mouse.midButton = []
        star_mouse.rightButton = []
        star_mouse.time = []
        star_mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        # keep track of which components have finished
        startComponents = [start_Lrectangle, start_Rrecntangle, start_L_image, start_R_image, star, star_mouse]
        for thisComponent in startComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "start" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *start_Lrectangle* updates
            
            # if start_Lrectangle is starting this frame...
            if start_Lrectangle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                start_Lrectangle.frameNStart = frameN  # exact frame index
                start_Lrectangle.tStart = t  # local t and not account for scr refresh
                start_Lrectangle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(start_Lrectangle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'start_Lrectangle.started')
                # update status
                start_Lrectangle.status = STARTED
                start_Lrectangle.setAutoDraw(True)
            
            # if start_Lrectangle is active this frame...
            if start_Lrectangle.status == STARTED:
                # update params
                pass
            
            # *start_Rrecntangle* updates
            
            # if start_Rrecntangle is starting this frame...
            if start_Rrecntangle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                start_Rrecntangle.frameNStart = frameN  # exact frame index
                start_Rrecntangle.tStart = t  # local t and not account for scr refresh
                start_Rrecntangle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(start_Rrecntangle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'start_Rrecntangle.started')
                # update status
                start_Rrecntangle.status = STARTED
                start_Rrecntangle.setAutoDraw(True)
            
            # if start_Rrecntangle is active this frame...
            if start_Rrecntangle.status == STARTED:
                # update params
                pass
            
            # *start_L_image* updates
            
            # if start_L_image is starting this frame...
            if start_L_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                start_L_image.frameNStart = frameN  # exact frame index
                start_L_image.tStart = t  # local t and not account for scr refresh
                start_L_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(start_L_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'start_L_image.started')
                # update status
                start_L_image.status = STARTED
                start_L_image.setAutoDraw(True)
            
            # if start_L_image is active this frame...
            if start_L_image.status == STARTED:
                # update params
                pass
            
            # *start_R_image* updates
            
            # if start_R_image is starting this frame...
            if start_R_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                start_R_image.frameNStart = frameN  # exact frame index
                start_R_image.tStart = t  # local t and not account for scr refresh
                start_R_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(start_R_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'start_R_image.started')
                # update status
                start_R_image.status = STARTED
                start_R_image.setAutoDraw(True)
            
            # if start_R_image is active this frame...
            if start_R_image.status == STARTED:
                # update params
                pass
            
            # *star* updates
            
            # if star is starting this frame...
            if star.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                star.frameNStart = frameN  # exact frame index
                star.tStart = t  # local t and not account for scr refresh
                star.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(star, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'star.started')
                # update status
                star.status = STARTED
                star.setAutoDraw(True)
            
            # if star is active this frame...
            if star.status == STARTED:
                # update params
                pass
            # *star_mouse* updates
            
            # if star_mouse is starting this frame...
            if star_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                star_mouse.frameNStart = frameN  # exact frame index
                star_mouse.tStart = t  # local t and not account for scr refresh
                star_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(star_mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('star_mouse.started', t)
                # update status
                star_mouse.status = STARTED
                star_mouse.mouseClock.reset()
                prevButtonState = star_mouse.getPressed()  # if button is down already this ISN'T a new click
            if star_mouse.status == STARTED:  # only update if started and not finished!
                buttons = star_mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(star, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(star_mouse):
                                gotValidClick = True
                                star_mouse.clicked_name.append(obj.name)
                        x, y = star_mouse.getPos()
                        star_mouse.x.append(x)
                        star_mouse.y.append(y)
                        buttons = star_mouse.getPressed()
                        star_mouse.leftButton.append(buttons[0])
                        star_mouse.midButton.append(buttons[1])
                        star_mouse.rightButton.append(buttons[2])
                        star_mouse.time.append(star_mouse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in startComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "start" ---
        for thisComponent in startComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('start.stopped', globalClock.getTime(format='float'))
        # store data for trials (TrialHandler)
        trials.addData('star_mouse.x', star_mouse.x)
        trials.addData('star_mouse.y', star_mouse.y)
        trials.addData('star_mouse.leftButton', star_mouse.leftButton)
        trials.addData('star_mouse.midButton', star_mouse.midButton)
        trials.addData('star_mouse.rightButton', star_mouse.rightButton)
        trials.addData('star_mouse.time', star_mouse.time)
        trials.addData('star_mouse.clicked_name', star_mouse.clicked_name)
        # the Routine "start" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial.started', globalClock.getTime(format='float'))
        left_image.setImage(l_image)
        right_image.setImage(right_image)
        trail_sound.setSound(target_sound, secs=1.59, hamming=True)
        trail_sound.setVolume(1.0, log=False)
        trail_sound.seek(0)
        # setup some python lists for storing info about the trial_mouse
        trial_mouse.x = []
        trial_mouse.y = []
        trial_mouse.leftButton = []
        trial_mouse.midButton = []
        trial_mouse.rightButton = []
        trial_mouse.time = []
        trial_mouse.clicked_name = []
        gotValidClick = False  # until a click is received
        # keep track of which components have finished
        trialComponents = [right_rectanle, left_rectangle, left_image, right_image, trail_sound, trial_mouse]
        for thisComponent in trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *right_rectanle* updates
            
            # if right_rectanle is starting this frame...
            if right_rectanle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                right_rectanle.frameNStart = frameN  # exact frame index
                right_rectanle.tStart = t  # local t and not account for scr refresh
                right_rectanle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(right_rectanle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_rectanle.started')
                # update status
                right_rectanle.status = STARTED
                right_rectanle.setAutoDraw(True)
            
            # if right_rectanle is active this frame...
            if right_rectanle.status == STARTED:
                # update params
                pass
            
            # *left_rectangle* updates
            
            # if left_rectangle is starting this frame...
            if left_rectangle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                left_rectangle.frameNStart = frameN  # exact frame index
                left_rectangle.tStart = t  # local t and not account for scr refresh
                left_rectangle.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(left_rectangle, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_rectangle.started')
                # update status
                left_rectangle.status = STARTED
                left_rectangle.setAutoDraw(True)
            
            # if left_rectangle is active this frame...
            if left_rectangle.status == STARTED:
                # update params
                pass
            
            # *left_image* updates
            
            # if left_image is starting this frame...
            if left_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                left_image.frameNStart = frameN  # exact frame index
                left_image.tStart = t  # local t and not account for scr refresh
                left_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(left_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_image.started')
                # update status
                left_image.status = STARTED
                left_image.setAutoDraw(True)
            
            # if left_image is active this frame...
            if left_image.status == STARTED:
                # update params
                pass
            
            # *right_image* updates
            
            # if right_image is starting this frame...
            if right_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                right_image.frameNStart = frameN  # exact frame index
                right_image.tStart = t  # local t and not account for scr refresh
                right_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(right_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_image.started')
                # update status
                right_image.status = STARTED
                right_image.setAutoDraw(True)
            
            # if right_image is active this frame...
            if right_image.status == STARTED:
                # update params
                pass
            
            # if trail_sound is starting this frame...
            if trail_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trail_sound.frameNStart = frameN  # exact frame index
                trail_sound.tStart = t  # local t and not account for scr refresh
                trail_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('trail_sound.started', tThisFlipGlobal)
                # update status
                trail_sound.status = STARTED
                trail_sound.play(when=win)  # sync with win flip
            
            # if trail_sound is stopping this frame...
            if trail_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > trail_sound.tStartRefresh + 1.59-frameTolerance:
                    # keep track of stop time/frame for later
                    trail_sound.tStop = t  # not accounting for scr refresh
                    trail_sound.tStopRefresh = tThisFlipGlobal  # on global time
                    trail_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trail_sound.stopped')
                    # update status
                    trail_sound.status = FINISHED
                    trail_sound.stop()
            # update trail_sound status according to whether it's playing
            if trail_sound.isPlaying:
                trail_sound.status = STARTED
            elif trail_sound.isFinished:
                trail_sound.status = FINISHED
            # *trial_mouse* updates
            
            # if trial_mouse is starting this frame...
            if trial_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trial_mouse.frameNStart = frameN  # exact frame index
                trial_mouse.tStart = t  # local t and not account for scr refresh
                trial_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('trial_mouse.started', t)
                # update status
                trial_mouse.status = STARTED
                trial_mouse.mouseClock.reset()
                prevButtonState = trial_mouse.getPressed()  # if button is down already this ISN'T a new click
            if trial_mouse.status == STARTED:  # only update if started and not finished!
                buttons = trial_mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames([left_image, right_image, left_rectangle, right_rectangle], namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(trial_mouse):
                                gotValidClick = True
                                trial_mouse.clicked_name.append(obj.name)
                        x, y = trial_mouse.getPos()
                        trial_mouse.x.append(x)
                        trial_mouse.y.append(y)
                        buttons = trial_mouse.getPressed()
                        trial_mouse.leftButton.append(buttons[0])
                        trial_mouse.midButton.append(buttons[1])
                        trial_mouse.rightButton.append(buttons[2])
                        trial_mouse.time.append(trial_mouse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial.stopped', globalClock.getTime(format='float'))
        trail_sound.pause()  # ensure sound has stopped at end of Routine
        # store data for trials (TrialHandler)
        trials.addData('trial_mouse.x', trial_mouse.x)
        trials.addData('trial_mouse.y', trial_mouse.y)
        trials.addData('trial_mouse.leftButton', trial_mouse.leftButton)
        trials.addData('trial_mouse.midButton', trial_mouse.midButton)
        trials.addData('trial_mouse.rightButton', trial_mouse.rightButton)
        trials.addData('trial_mouse.time', trial_mouse.time)
        trials.addData('trial_mouse.clicked_name', trial_mouse.clicked_name)
        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials'
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
