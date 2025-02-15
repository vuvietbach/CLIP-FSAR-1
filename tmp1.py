_base_ = ['../ssv2/clipfsar_ssv2original.py']
TRAIN = dict(
    CLASS_NAME = ['air drumming', 'arm wrestling', 'beatboxing', 'biking through snow', 'blowing glass', 'blowing out candles', 'bowling', 'breakdancing', 'bungee jumping', 'catching or throwing baseball', 'cheerleading', 'cleaning floor', 'contact juggling', 'cooking chicken', 'country line dancing', 'curling hair', 'deadlifting', 'doing nails', 'dribbling basketball', 'driving tractor', 'drop kicking', 'dying hair', 'eating burger', 'feeding birds', 'giving or receiving award', 'hopscotch', 'jetskiing', 'jumping into pool', 'laughing', 'making snowman', 'massaging back', 'mowing lawn', 'opening bottle', 'playing accordion', 'playing badminton', 'playing basketball', 'playing didgeridoo', 'playing ice hockey', 'playing keyboard', 'playing ukulele', 'playing xylophone', 'presenting weather forecast', 'punching bag', 'pushing cart', 'reading book', 'riding unicycle', 'shaking head', 'sharpening pencil', 'shaving head', 'shot put', 'shuffling cards', 'slacklining', 'sled dog racing', 'snowboarding', 'somersaulting', 'squat', 'surfing crowd', 'trapezing', 'using computer', 'washing dishes', 'washing hands', 'water skiing', 'waxing legs', 'weaving basket']
)
TEST = dict(
    CLASS_NAME = ['blasting sand',  'busking',  'cutting watermelon',  'dancing ballet', 'dancing charleston',  'dancing macarena',  'diving cliff', 'filling eyebrows', 'folding paper',  'hula hooping', 'hurling (sport)',  'ice skating',  'paragliding', 'playing drums',  'playing monopoly', 'playing trumpet', 'pushing car', 'riding elephant',  'shearing sheep', 'side kick', 'stretching arm', 'tap dancing', 'throwing axe',  'unboxing']
)
VAL = dict(
    CLASS_NAME = ["baking cookies", "crossing river", "dunking basketball", "feeding fish", "flying kite", "high kick", "javelin throw", "playing trombone", "scuba diving", "skateboarding", "ski jumping", "trimming or shaving beard"]
)

DATA = dict(
    DATA_ROOT_DIR="data/kinetics/videos",
    ANNO_DIR = 'data/kinetics/annotations',
    VIDEO_FORMAT='',
)