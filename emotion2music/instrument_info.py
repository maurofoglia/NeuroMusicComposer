"""
Instrument Information Database.
Contains the practical range, optimal clef, and musical idiom for 128 standard MIDI instruments.
"""

INSTRUMENT_INFO = {
    # --- PIANOS (1-8) ---
    "1 Piano 1": {"range": "[A0-C8]", "clef": "G", "idiom": "Use expressive piano phrasing, syncopations, and chords. Feel free to use ledger lines."},
    "2 Piano 2": {"range": "[A0-C8]", "clef": "G", "idiom": "Bright acoustic piano. Good for pop/rock rhythmic comping."},
    "3 Piano 3": {"range": "[A0-C8]", "clef": "G", "idiom": "Electric grand style. Punchy and aggressive chord stabs."},
    "4 Honky-tonk": {"range": "[A0-C8]", "clef": "G", "idiom": "Ragtime/saloon style. Fast 1/8th note runs and stride bass."},
    "5 E. Piano 1": {"range": "[E1-C7]", "clef": "G", "idiom": "Rhodes style. Warm jazzy comping, 7th/9th chords, and groovy rhythms."},
    "6 E. Piano 2": {"range": "[E1-C7]", "clef": "G", "idiom": "FM synth piano (DX7 style). Bright, 80s pop ballads and arpeggios."},
    "7 Harpsichord": {"range": "[F1-F6]", "clef": "G", "idiom": "Continuous contrapuntal motion (baroque style). Trills and mordents. NO velocity dynamics."},
    "8 Clav.": {"range": "[E1-F6]", "clef": "G", "idiom": "Extremely syncopated, 1/16th note funk patterns. Staccato."},

    # --- CHROMATIC PERCUSSION (9-16) ---
    "9 Celesta": {"range": "[C3-C7]", "clef": "G", "idiom": "Fairy-tale, bell-like melodies. Delicate arpeggios."},
    "10 Glockenspiel": {"range": "[G3-C8]", "clef": "G", "idiom": "Piercing, high-pitched metallic runs. Keep it sparse."},
    "11 Music Box": {"range": "[C4-C7]", "clef": "G", "idiom": "Mechanical, rigid but sweet lullaby melodies."},
    "12 Vibraphone": {"range": "[F3-F6]", "clef": "G", "idiom": "Jazz idiom. 4-mallet chordal comping and fluid solo lines."},
    "13 Marimba": {"range": "[A2-C7]", "clef": "G", "idiom": "Fast, rhythmic, woody arpeggios and ostinatos. Tropical feel."},
    "14 Xylophone": {"range": "[F3-C8]", "clef": "G", "idiom": "Dry, highly rhythmic and playful staccato lines."},
    "15 Tubular-bell": {"range": "[C4-F5]", "clef": "G", "idiom": "Dramatic, slow ringing tolls. Do not play fast."},
    "16 Santur": {"range": "[C3-C6]", "clef": "G", "idiom": "Middle-Eastern hammered dulcimer. Fast tremolos and scale runs."},

    # --- ORGANS (17-24) ---
    "17 Organ 1": {"range": "[C2-C7]", "clef": "G", "idiom": "Hammond B3 style. Bluesy licks, sustained chords, and glissandos."},
    "18 Organ 2": {"range": "[C2-C7]", "clef": "G", "idiom": "Percussive rock/jazz organ. Fast runs and groovy rhythmic stabs."},
    "19 Organ 3": {"range": "[C2-C7]", "clef": "G", "idiom": "Rock organ. Heavy sustained chords for foundational harmony."},
    "20 Church Org. 1": {"range": "[C2-C7]", "clef": "G", "idiom": "Massive baroque chords. Imposing contrapuntal lines."},
    "21 Reed Organ": {"range": "[F2-F6]", "clef": "G", "idiom": "Slow, pumping acoustic chords. Melancholy feel."},
    "22 Accordion Fr": {"range": "[F2-A5]", "clef": "G", "idiom": "Folk/Musette waltz patterns. Oom-pah bass and chord style."},
    "23 Harmonica": {"range": "[C3-C6]", "clef": "G", "idiom": "Bluesy bends, soulful folk melodies. Single notes mainly."},
    "24 Bandneon": {"range": "[C3-A5]", "clef": "G", "idiom": "Tango style. Staccato rhythmic chords mixed with passionate melodies."},

    # --- GUITARS (25-32) ---
    "25 Nylon-str. Gt": {"range": "[E2-A5]", "clef": "G", "idiom": "Classical/Flamenco fingerpicking patterns and arpeggiated chords."},
    "26 Steel-Str. Gt": {"range": "[E2-C6]", "clef": "G", "idiom": "Folk pop strumming patterns and bright fingerpicking."},
    "27 Jazz Gt.": {"range": "[E2-C6]", "clef": "G", "idiom": "Warm, complex jazz chords (7ths, 9ths) and smooth syncopated lines."},
    "28 Clean Gt.": {"range": "[E2-D6]", "clef": "G", "idiom": "Funky 1/16th note rhythm strumming or chorus-heavy pop arpeggios."},
    "29 Muted Gt.": {"range": "[E2-D6]", "clef": "G", "idiom": "Very percussive, tight 1/16th note funk/pop rhythmic lines."},
    "30 Overdrive Gt": {"range": "[E2-D6]", "clef": "G", "idiom": "Rock power chords (fifths) and bluesy, slightly dirty riffs."},
    "31 DistortionGt": {"range": "[E2-D6]", "clef": "G", "idiom": "Heavy metal riffs, palm muting, and aggressive fast solos."},
    "32 Gt. Harmonics": {"range": "[E3-E6]", "clef": "G", "idiom": "Ethereal, bell-like acoustic pings. Sparse arpeggios."},

    # --- BASSES (33-40) ---
    "33 Acoustic Bs.": {"range": "[E1-G3]", "clef": "F", "idiom": "STRICTLY single line. Upright walking bass lines, strong swing feel."},
    "34 Fingered Bs.": {"range": "[E1-G3]", "clef": "F", "idiom": "STRICTLY single line. Groovy, rhythmic R&B/Pop walking lines."},
    "35 Picked Bs.": {"range": "[E1-G3]", "clef": "F", "idiom": "STRICTLY single line. Driving, consistent and punchy 1/8th note rock patterns."},
    "36 Fretless Bs.": {"range": "[E1-G3]", "clef": "F", "idiom": "STRICTLY single line. Expressive, sliding fusion/jazz melodies and basslines."},
    "37 Slap Bs. 1": {"range": "[E1-G3]", "clef": "F", "idiom": "Highly syncopated 1/16th note funk thumb slaps and pops."},
    "38 Slap Bs. 2": {"range": "[E1-G3]", "clef": "F", "idiom": "Aggressive funk slaps with lots of ghost notes."},
    "39 Synth Bass 1": {"range": "[C1-C3]", "clef": "F", "idiom": "Punchy, electronic sub-bass. Repetitive 1/16th synthwave patterns."},
    "40 Synth Bass 2": {"range": "[C1-C3]", "clef": "F", "idiom": "Acid/analog bass. Uses glides and resonant filter sweeps."},

    # --- STRINGS (41-48) ---
    "41 Violin": {"range": "[G3-C7]", "clef": "G", "idiom": "STRICTLY single melodic line. Long bowed notes and fast runs."},
    "42 Viola": {"range": "[C3-E6]", "clef": "G", "idiom": "Single line only. Warm inner harmonies and counter-melodies."},
    "43 Cello": {"range": "[C2-A5]", "clef": "F", "idiom": "Single line only. Strong bowed basslines and expressive tenor melodies."},
    "44 Contrabass": {"range": "[E1-G3]", "clef": "F", "idiom": "Single line only. Deep, steady harmonic foundation. Avoid fast 1/16 notes."},
    "45 Tremolo Str": {"range": "[G3-C7]", "clef": "G", "idiom": "Fast, nervous bowing. Use for suspenseful or climatic sustained chords."},
    "46 PizzicatoStr": {"range": "[C2-C7]", "clef": "G", "idiom": "Short, plucked strings. Play playful, staccato 1/8th note lines."},
    "47 Harp": {"range": "[Cb1-G#7]", "clef": "G", "idiom": "Sweeping glissandos and wide, rolled arpeggios."},
    "48 Timpani": {"range": "[C2-C3]", "clef": "F", "idiom": "Orchestral rolls and powerful percussive hits. Single notes or 4ths/5ths."},

    # --- ENSEMBLE (49-56) ---
    "49 Strings": {"range": "[C2-C7]", "clef": "G", "idiom": "Full orchestral string section. Lush chords and sweeping melodies."},
    "50 Slow Strings": {"range": "[C2-C7]", "clef": "G", "idiom": "Very slow attack. Use ONLY for long, sustained whole-note chords."},
    "51 Syn. Strings1": {"range": "[C2-C7]", "clef": "G", "idiom": "80s analog string machine. Warm, sustained pop/disco pads."},
    "52 Syn. Strings2": {"range": "[C2-C7]", "clef": "G", "idiom": "Brighter synth strings. Good for rhythmic chord stabs or pads."},
    "53 Choir Aahs": {"range": "[F2-C6]", "clef": "G", "idiom": "Human choir. Majestic, slow-moving block chords."},
    "54 Voice Oohs": {"range": "[F2-C6]", "clef": "G", "idiom": "Softer choir. Ethereal, gentle sustained background harmonies."},
    "55 SynVox": {"range": "[C2-C7]", "clef": "G", "idiom": "Artificial vocal synth. 90s dance pads and robotic choirs."},
    "56 OrchestraHit": {"range": "[C2-C7]", "clef": "G", "idiom": "Massive staccato orchestral tutti. Use for dramatic rhythmic accents."},

    # --- BRASS (57-64) ---
    "57 Trumpet": {"range": "[E3-C6]", "clef": "G", "idiom": "Single line only. Bright, heroic fanfare motifs. Staccato and accents."},
    "58 Trombone": {"range": "[E2-F4]", "clef": "F", "idiom": "Single line only. Powerful bass notes. Slides (legato) and strong impacts."},
    "59 Tuba": {"range": "[D1-F3]", "clef": "F", "idiom": "Single line only. Deep, resonant bass. Steady quarter notes, no fast leaps."},
    "60 MutedTrumpet": {"range": "[E3-C6]", "clef": "G", "idiom": "Single line. Jazzy, nasal, Miles Davis style. Staccato swing lines."},
    "61 French Horn": {"range": "[F2-F5]", "clef": "F", "idiom": "Single line only. Noble and mellow. Soaring, sustained melodies."},
    "62 Brass 1": {"range": "[F2-C6]", "clef": "G", "idiom": "Full brass section. Powerful, punchy chordal stabs and swells."},
    "63 Synth Brass1": {"range": "[C2-C6]", "clef": "G", "idiom": "80s synth brass (e.g., Jump). Huge, bright pop chords."},
    "64 Synth Brass2": {"range": "[C2-C6]", "clef": "G", "idiom": "Warmer, analog synth brass. Good for slower electronic pads."},

    # --- WOODWINDS (65-72) ---
    "65 Soprano sax": {"range": "[Ab3-E6]", "clef": "G", "idiom": "Single line. Smooth jazz melodies, highly ornamented."},
    "66 Alto sax": {"range": "[Db3-A5]", "clef": "G", "idiom": "Single line. Expressive, jazzy, with syncopated rhythms and bends."},
    "67 Tenor sax": {"range": "[Ab2-E5]", "clef": "G", "idiom": "Single line. Breathy, bluesy, and powerful rock/jazz solos."},
    "68 Baritone sax": {"range": "[C2-A4]", "clef": "F", "idiom": "Single line. Gritty, rhythmic bass lines or honking R&B stabs."},
    "69 Oboe": {"range": "[Bb3-G5]", "clef": "G", "idiom": "Single line. Sweet, melancholic, piercing melodies. Legato phrasing."},
    "70 English Horn": {"range": "[E3-C5]", "clef": "G", "idiom": "Single line. Mournful, pastoral. Very expressive and slow."},
    "71 Bassoon": {"range": "[Bb1-Eb4]", "clef": "F", "idiom": "Single line. Quirky staccato bass lines or lyrical tenor melodies."},
    "72 Clarinet": {"range": "[D3-C6]", "clef": "G", "idiom": "Single line. Very agile, arpeggiated runs and warm sustained notes."},

    # --- PIPES/FLUTES (73-80) ---
    "73 Piccolo": {"range": "[D5-C8]", "clef": "G", "idiom": "Single line. Extremely high, piercing, fast trills and runs."},
    "74 Flute": {"range": "[C4-C7]", "clef": "G", "idiom": "Single line. Light, airy, fast. Frequent leaps and breathy phrases."},
    "75 Recorder": {"range": "[F4-G6]", "clef": "G", "idiom": "Single line. Medieval/Renaissance feel, simple and pure melodies."},
    "76 Pan flute": {"range": "[C4-C7]", "clef": "G", "idiom": "Single line. Andean/World feel. Lots of breath and pitch scoops."},
    "77 Bottle Blow": {"range": "[C4-C6]", "clef": "G", "idiom": "Single line. Hollow, resonant, staccato rhythmic blowing."},
    "78 Shakuhachi": {"range": "[D4-G6]", "clef": "G", "idiom": "Single line. Japanese bamboo flute. Slow, highly bent and expressive notes."},
    "79 Whistle": {"range": "[C5-C7]", "clef": "G", "idiom": "Single line. Cheerful, simple folk melodies and slides."},
    "80 Ocarina": {"range": "[C4-F6]", "clef": "G", "idiom": "Single line. Sweet, round tone. Simple, flowing melodies."},

    # --- SYNTH LEADS (81-88) ---
    "81 Square Wave": {"range": "[C3-C7]", "clef": "G", "idiom": "Single line monophonic. Fast, energetic, chiptune/8-bit arpeggios."},
    "82 Saw Wave": {"range": "[C3-C7]", "clef": "G", "idiom": "Aggressive, cutting lead. Pitch bends (ties) and syncopation. Trance style."},
    "83 Syn. Calliope": {"range": "[C3-C7]", "clef": "G", "idiom": "Bouncy, circus-like synth melodies. Staccato."},
    "84 Chiffer Lead": {"range": "[C3-C7]", "clef": "G", "idiom": "Breathy attack synth lead. Good for new-age melodies."},
    "85 Charang": {"range": "[C3-C7]", "clef": "G", "idiom": "Distorted, syncopated synth guitar-like lead."},
    "86 Solo Vox": {"range": "[C3-C7]", "clef": "G", "idiom": "Vocal-like synth lead. Smooth portamento and legato phrasing."},
    "87 5th Saw Wave": {"range": "[C3-C7]", "clef": "G", "idiom": "Techno lead playing parallel fifths. Fast, rhythmic hooks."},
    "88 Bass & Lead": {"range": "[C1-C5]", "clef": "F", "idiom": "Fat analog bass/lead. Funky 1/16th runs or heavy sub drops."},

    # --- SYNTH PADS (89-96) ---
    "89 Fantasia": {"range": "[C3-C7]", "clef": "G", "idiom": "Sparkling, evolving digital pad. Slow arpeggios or sustained chords."},
    "90 Warm Pad": {"range": "[C2-C6]", "clef": "G", "idiom": "Long, sustained chordal textures (1/2 or whole notes). Very little rhythm."},
    "91 Polysynth": {"range": "[C2-C7]", "clef": "G", "idiom": "Thick, 80s pop chords and bright, brassy rhythmic stabs."},
    "92 Space Voice": {"range": "[C2-C6]", "clef": "G", "idiom": "Ethereal, dark vocal pad. Slow, cinematic chord progressions."},
    "93 Bowed Glass": {"range": "[C3-C7]", "clef": "G", "idiom": "High, crystalline atmospheric pad. Ambient sustained notes."},
    "94 Metal Pad": {"range": "[C2-C6]", "clef": "G", "idiom": "Metallic, detuned ambient pad. Industrial/sci-fi chords."},
    "95 Halo Pad": {"range": "[C2-C6]", "clef": "G", "idiom": "Angelic, resonant pad. Keep movements slow and majestic."},
    "96 Sweep Pad": {"range": "[C2-C6]", "clef": "G", "idiom": "Pad with a slow filter sweep. Requires long note values to evolve."},

    # --- SYNTH EFFECTS (97-104) ---
    "97 Ice Rain": {"range": "[C4-C7]", "clef": "G", "idiom": "Randomized, high-pitched bell textures. Sparse, ambient drops."},
    "98 Soundtrack": {"range": "[C2-C6]", "clef": "G", "idiom": "Huge, cinematic evolving chordal soundscape."},
    "99 Crystal": {"range": "[C4-C7]", "clef": "G", "idiom": "Sharp, delicate synth bells. Short repetitive arpeggios."},
    "100 Atmosphere": {"range": "[C2-C6]", "clef": "G", "idiom": "Dark, moody drone pad. Very slow changes."},
    "101 Brightness": {"range": "[C3-C7]", "clef": "G", "idiom": "Shimmering, bright attack pad. Good for uplifting chords."},
    "102 Goblin": {"range": "[C2-C6]", "clef": "G", "idiom": "Weird, modulating, unsettling synth texture."},
    "103 Echo Drops": {"range": "[C3-C7]", "clef": "G", "idiom": "Rhythmic, echoing synth plucks. Good for spatial arpeggios."},
    "104 Star Theme": {"range": "[C3-C7]", "clef": "G", "idiom": "Sci-fi, spacey evolving pad/lead. Sweeping melodies."},

    # --- ETHNIC INSTRUMENTS (105-112) ---
    "105 Sitar": {"range": "[C3-C6]", "clef": "G", "idiom": "Indian string. Drone notes and highly bent, microtonal melodies."},
    "106 Banjo": {"range": "[C3-C6]", "clef": "G", "idiom": "Fast bluegrass fingerpicking rolls. Very staccato and twangy."},
    "107 Shamisen": {"range": "[C3-C6]", "clef": "G", "idiom": "Japanese plucked string. Snappy, rhythmic, and percussive single lines."},
    "108 Koto": {"range": "[G2-C6]", "clef": "G", "idiom": "Japanese zither. Pentatonic glissandos and delicate plucked melodies."},
    "109 Kalimba": {"range": "[C3-C6]", "clef": "G", "idiom": "African thumb piano. Polyrhythmic, repetitive, and gentle plucks."},
    "110 Bag Pipe": {"range": "[G3-G5]", "clef": "G", "idiom": "Continuous drone note beneath a rigid, highly ornamented melody."},
    "111 Fiddle": {"range": "[G3-C7]", "clef": "G", "idiom": "Folk violin. Fast, scraping, rhythmic bowing and double stops."},
    "112 Shanai": {"range": "[C4-C6]", "clef": "G", "idiom": "Piercing, nasal Indian wind. Loud, continuous slurred melodies."},

    # --- PERCUSSION (113-120) ---
    "113 Tinkle Bell": {"range": "[C5-C7]", "clef": "G", "idiom": "High, metallic jingling. Rhythmic accents."},
    "114 Agogo": {"range": "[C4-C5]", "clef": "G", "idiom": "Latin metal bells. Highly syncopated samba/salsa rhythms."},
    "115 Steel Drums": {"range": "[C3-C6]", "clef": "G", "idiom": "Caribbean calypso melodies and rolling chordal comping."},
    "116 Woodblock": {"range": "[C4-C5]", "clef": "G", "idiom": "Dry, clicky rhythmic timekeeping. Often off-beats."},
    "117 Taiko": {"range": "[C2-C4]", "clef": "F", "idiom": "Massive Japanese drums. Heavy, dramatic rhythmic impacts."},
    "118 Melo Tom 1": {"range": "[C2-C4]", "clef": "F", "idiom": "Tonal drum fills. Rhythmic patterns crossing different pitches."},
    "119 Synth Drum": {"range": "[C2-C4]", "clef": "F", "idiom": "80s electronic tom-toms (e.g., Simmons). Big descending fills."},
    "120 Reverse: Cym.": {"range": "[C3-C4]", "clef": "G", "idiom": "A cymbal playing backwards. Use on beat 4 leading into a downbeat."},

    # --- SOUND EFFECTS (121-128) ---
    "121 Gt. FretNoise": {"range": "[C3-C5]", "clef": "G", "idiom": "Non-pitched guitar squeaks. Use sparingly for realism in rests."},
    "122 Fl. Keyclick": {"range": "[C3-C5]", "clef": "G", "idiom": "Mechanical breath/click noises. Very percussive."},
    "123 Seashore": {"range": "[C3-C5]", "clef": "G", "idiom": "White noise wash. Hold for long durations as ambient background."},
    "124 Bird": {"range": "[C5-C7]", "clef": "G", "idiom": "Chirping sounds. Short, random high-pitched bursts."},
    "125 Telephone 1": {"range": "[C4-C5]", "clef": "G", "idiom": "Ringing bell. Rhythmic, repetitive alarm-like trills."},
    "126 Helicopter": {"range": "[C2-C4]", "clef": "F", "idiom": "Low frequency rhythmic thumping. Hold a single note."},
    "127 Applause": {"range": "[C3-C5]", "clef": "G", "idiom": "Crowd noise. Trigger at the end of a piece."},
    "128 Gun Shot": {"range": "[C2-C4]", "clef": "F", "idiom": "Sharp, transient burst. Use for heavy, singular percussive downbeats."}
}