import os
import random
import time

import torch
from compel import Compel
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

output_dir = "test/"
sample_size = 5

# fish
species_data = [
    {
        "species": "Spotted Drum",
        "main_color": "black",
        "secondary_color": "white stripes and spots",
        "body_shape": "deep and laterally compressed with an elongated dorsal fin",
        "num_v_stripes": 2,
        "num_h_stripes": 2,
        "num_spots": 6,
        "mouth_angle": "small and slightly downturned",
        "habitat": "coral reefs and lagoons",
        "behavioral_traits": {
            "solitary": 1,
            "territorial": 1,
            "diurnal": 1
        },
        "group_behavior": "frequently found in small groups or loose aggregations, often near the seabed in shallow waters",
        "fins": "elongated dorsal fin, rounded pectoral fins, slightly forked tail fin",
        "scale_texture": "smooth and glossy with fine scales",
        "eye_characteristics": "large and round with a dark pupil and silver ring",
        "life_stage": "adult",
        "lighting_effects": "natural reef lighting with slight iridescence on scales",
        "avoid_features": "do not exaggerate dorsal fin length, no excessive brightness"
    },  # Spotted Drum
    {
        "species": "Yellowtail Damselfish",
        "main_color": "blue",
        "secondary_color": "yellow tail",
        "body_shape": "oval-shaped and compact",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "small and slightly protruded",
        "habitat": "shallow coral reefs and rocky areas",
        "behavioral_traits": {
            "territorial": 1,
            "aggressive": 1,
            "social": 1
        },
        "group_behavior": "often seen in small schools or defending territories near coral reefs",
        "fins": "rounded dorsal fin, slightly forked caudal fin",
        "scale_texture": "small, rough-textured scales",
        "eye_characteristics": "small and round with a dark iris",
        "life_stage": "adult",
        "lighting_effects": "bright tropical lighting, shimmering in sunlight",
        "avoid_features": "do not exaggerate yellow tail size"
    },  # Yellowtail Damselfish
    {
        "species": "Sergeant Major",
        "main_color": "silvery gold/yellow",
        "secondary_color": "black vertical stripes",
        "body_shape": "flat and elongated",
        "num_v_stripes": 5,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "small and rounded",
        "habitat": "coral reefs, lagoons, and coastal areas",
        "behavioral_traits": {
            "active": 1,
            "schooling": 1,
            "territorial": 1
        },
        "group_behavior": "swims in schools of 10-50 individuals",
        "fins": "pointed dorsal fin, forked tail fin",
        "scale_texture": "noticeable overlapping scales with a smooth surface",
        "eye_characteristics": "moderate-sized eyes with a yellow ring around the pupil",
        "life_stage": "adult",
        "lighting_effects": "sunlit waters creating a shimmering effect",
        "avoid_features": "do not distort the black stripe spacing"
    },  # Sergent major
    {
        "species": "Queen Angelfish",
        "main_color": "vibrant blue",
        "secondary_color": "yellow with blue-green gradients",
        "body_shape": "oval and laterally compressed",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "small and slightly protruding",
        "habitat": "coral reefs and rocky environments",
        "behavioral_traits": {
            "grazing": 1,
            "peaceful": 1,
            "social": 1
        },
        "group_behavior": "usually solitary or seen in pairs",
        "fins": "elongated dorsal fin, rounded pectoral fins, and a slightly crescent-shaped tail fin",
        "scale_texture": "small, smooth, and iridescent scales",
        "eye_characteristics": "large and round with a striking blue ring around the pupil",
        "lighting_effects": "soft underwater glow highlighting its iridescent scales",
        "avoid_features": "do not exaggerate yellow gradients; keep body shape natural"
    },  # Queen Angelfish
    {
        "species": "Squirrelfish",
        "main_color": "reddish-pink",
        "secondary_color": "silvery-white",
        "body_shape": "elongated and slightly compressed",
        "num_v_stripes": 0,
        "num_h_stripes": 3,
        "num_spots": 0,
        "mouth_angle": "slightly downturned",
        "habitat": "coral reefs, rocky areas, and deep caves",
        "behavioral_traits": {
            "nocturnal": 1,
            "territorial": 1,
            "social": 1
        },
        "group_behavior": "usually seen in small groups or alone, hiding in crevices during the day and becoming active at night",
        "fins": "large spiny dorsal fin, forked tail fin, and prominent pectoral fins",
        "scale_texture": "rough-textured with a slightly metallic sheen",
        "eye_characteristics": "large, round, and dark to enhance night vision",
        "lighting_effects": "low-light setting to emphasize nocturnal adaptation",
        "avoid_features": "do not make eyes too small; keep body proportions accurate"
    },  # Squirrelfish
    {
        "species": "Bluehead Wrasse",
        "main_color": "blue",
        "secondary_color": "green",
        "body_shape": "slender and elongated",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "slightly forward-facing and protrusible",
        "habitat": "coral reefs, sandy areas, and rocky shores",
        "behavioral_traits": {
            "cleaning": 1,
            "aggressive": 1,
            "social": 1
        },
        "group_behavior": "frequently seen in large schools, particularly during feeding or spawning activities",
        "fins": "long dorsal fin running along the back, pointed pectoral fins, and a fan-like caudal fin",
        "scale_texture": "smooth and streamlined with a slightly glossy finish",
        "eye_characteristics": "small and round with a bright reflective quality",
        "lighting_effects": "natural sunlight filtering through shallow waters",
        "avoid_features": "do not overemphasize green color; keep blue head distinct"
    },  # Bluehead Wrasse
    {
        "species": "Adult Stoplight Parrotfish",
        "main_color": "green",
        "secondary_color": "yellow with red patches",
        "body_shape": "robust and oval-shaped",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "forward-facing with a beak-like structure for grazing",
        "habitat": "shallow coral reefs and rocky areas",
        "behavioral_traits": {
            "grazing": 1,
            "territorial": 1,
            "social": 1
        },
        "group_behavior": "commonly observed in small mixed-gender groups, feeding on algae on the reef",
        "fins": "broad dorsal fin, rounded pectoral fins, and a crescent-shaped tail fin",
        "scale_texture": "large, overlapping scales with a slightly rough texture",
        "eye_characteristics": "medium-sized, round, with a reflective greenish tint",
        "lighting_effects": "soft dappled light filtering through shallow reef waters",
        "avoid_features": "do not exaggerate red patches; maintain realistic beak-like mouth structure"
    },  # Adult Stoplight Parrotfish
    {
        "species": "Blue Tang",
        "main_color": "vibrant blue",
        "secondary_color": "yellow tail",
        "body_shape": "oval and laterally compressed",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "small and slightly protruding",
        "habitat": "coral reefs, lagoons, and rocky shores",
        "behavioral_traits": {
            "schooling": 1,
            "grazing": 1,
            "peaceful": 1
        },
        "group_behavior": "commonly seen in large schools, often grazing on algae together, creating a shimmering effect in the water",
        "fins": "elongated dorsal fin, pointed pectoral fins, and a crescent-shaped caudal fin",
        "scale_texture": "smooth and slightly iridescent, with a velvety appearance",
        "eye_characteristics": "large, round, with a dark pupil contrasting against a lighter iris",
        "lighting_effects": "sunlight filtering through the water, creating a vibrant glow on its blue body",
        "avoid_features": "do not make body too elongated; keep the tail distinctively yellow"
    },  # Blue Tang
    {
        "species": "Southern Stingray",
        "main_color": "greyish-brown",
        "secondary_color": "white ventral surface",
        "body_shape": "flattened, disc-shaped, with a whip-like tail",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "small and slightly downward-facing",
        "habitat": "shallow coastal waters, sandy or muddy seafloor",
        "behavioral_traits": {
            "bottom-dwelling": 1,
            "slow movements": 1,
            "burrowing": 1
        },
        "group_behavior": "usually solitary or in small groups, can congregate during mating season",
        "fins": "broad pectoral fins forming a diamond-like shape, with a long, whip-like tail",
        "scale_texture": "smooth, leathery skin with a subtle grainy texture",
        "eye_characteristics": "small, oval-shaped, positioned on the top of the head",
        "lighting_effects": "soft, diffused light creating a sandy seabed ambiance",
        "avoid_features": "do not overemphasize ridges; keep the tail realistically proportioned"
    },  # Southern Stingray
    {
        "species": "Yellow Stingray",
        "main_color": "yellowish-brown",
        "secondary_color": "paler underside",
        "body_shape": "flattened, oval-shaped, with a whip-like tail",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "small and downward-facing",
        "habitat": "shallow coastal waters, sandy or seagrass beds",
        "behavioral_traits": {
            "bottom-dwelling": 1,
            "slow movements": 1,
            "burrowing": 1
        },
        "group_behavior": "usually solitary, but may form small groups when feeding",
        "fins": "rounded pectoral fins forming an oval disc, with a long, slender tail",
        "scale_texture": "smooth, slightly rough to the touch, with a camouflage-like pattern",
        "eye_characteristics": "small and oval, positioned on the dorsal side",
        "lighting_effects": "gentle, natural light filtering through shallow water",
        "avoid_features": "do not make the body too circular; maintain proper proportions"
    },  # Yellow Stingray
    {
        "species": "Nassau Grouper",
        "main_color": "grayish-brown",
        "secondary_color": "pale yellowish-brown or whitish",
        "body_shape": "laterally compressed, elongated body",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": 6,
        "mouth_angle": "large, forward-facing",
        "habitat": "shallow coastal waters, reefs, and seagrass beds",
        "behavioral_traits": {
            "territorial": 1,
            "reef-dwelling": 1,
            "ambush predator": 1
        },
        "group_behavior": "social, often seen in small groups or pairs",
        "fins": "broad dorsal fin with spines, rounded pectoral fins, and a large fan-shaped tail",
        "scale_texture": "rough and thick, with visible ridges",
        "eye_characteristics": "medium-sized, round, with a golden-brown tint",
        "lighting_effects": "subtle reef shadows and dappled sunlight through the water",
        "avoid_features": "do not make the body too elongated; keep the mouth proportionally large"
    },  # Nassau Grouper
    {
        "species": "Smooth Trunkfish",
        "main_color": "black with white spots",
        "secondary_color": "yellowish on the fins",
        "body_shape": "box-like and rounded",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": "many small white spots covering the body",
        "mouth_angle": "small and slightly forward-facing",
        "habitat": "coral reefs, rocky outcrops, and sandy bottoms, typically in shallow waters",
        "behavioral_traits": {
            "slow-moving": 1,
            "reef-dwelling": 1,
            "hovering swimmer": 1
        },
        "group_behavior": "usually solitary but occasionally forms small groups near reef structures",
        "fins": "small dorsal fin, rounded pectoral fins used for propulsion",
        "scale_texture": "smooth, hard, armor-like skin",
        "eye_characteristics": "large, round, slightly protruding with a dark pupil",
        "lighting_effects": "soft reflections on its armored body with gentle coral reef shadows",
        "avoid_features": "do not make the spots too large; keep the body compact and box-like"
    },  # Smooth Trunkfish
    {
        "species": "Spotted Moray Eel",
        "main_color": "brownish-yellow with dark brown spots",
        "secondary_color": "white highlights around the head and body",
        "body_shape": "elongated and cylindrical",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": "numerous dark spots scattered across the body",
        "mouth_angle": "wide and slightly upward-facing, with visible sharp teeth",
        "habitat": "coral reefs, rocky crevices, and sandy bottoms in shallow to mid-depth waters",
        "behavioral_traits": {
            "ambush predator": 1,
            "nocturnal": 1,
            "reef-dwelling": 1
        },
        "group_behavior": "typically solitary, occasionally seen sharing a crevice with other eels",
        "fins": "continuous dorsal and anal fins running along the body",
        "scale_texture": "smooth, almost scaleless appearance",
        "eye_characteristics": "small, round, positioned slightly forward",
        "lighting_effects": "dark reef shadows with occasional bright highlights on its spots",
        "avoid_features": "do not over-exaggerate the teeth; maintain natural spot distribution"
    },  # Spotted Moray Eel
    {
        "species": "Green Moray Eel",
        "main_color": "olive green to dark green, often appearing solid-colored",
        "secondary_color": "subtle yellowish tint near the fins and underside",
        "body_shape": "elongated and muscular, cylindrical body",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "wide and slightly downward-facing, sharp teeth often visible",
        "habitat": "coral reefs, wrecks, and rocky crevices, often at depths ranging from shallow waters to 40 meters",
        "behavioral_traits": {
            "territorial": 1,
            "ambush predator": 1,
            "reef-dwelling": 1
        },
        "group_behavior": "predominantly solitary, rarely seen with other eels unless in a shared crevice",
        "fins": "continuous dorsal and anal fins running the length of the body",
        "scale_texture": "smooth and leathery, with a slightly slimy sheen",
        "eye_characteristics": "small, round, and dark, often blending with the face",
        "lighting_effects": "moody deep reef lighting with subtle shadows",
        "avoid_features": "do not add patterns; keep the coloration uniform"
    },  # Green Moray Eel
    {
        "species": "Atlantic Spadefish",
        "main_color": "silvery white",
        "secondary_color": "dark gray or black vertical bars",
        "body_shape": "deep, flattened, and rounded",
        "num_v_stripes": 4,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "small and forward-facing",
        "habitat": "coastal waters, around piers, wrecks, and reefs",
        "behavioral_traits": {
            "schooling": 1,
            "curious": 1,
            "reef-dwelling": 1
        },
        "group_behavior": "frequently seen in large groups or schools, swimming slowly and cohesively",
        "fins": "long dorsal and anal fins that taper towards the tail, rounded pectoral fins",
        "scale_texture": "smooth, reflective, with a slightly metallic sheen",
        "eye_characteristics": "large, round, positioned centrally on the head",
        "lighting_effects": "sunlight reflections creating a shimmering effect on the silver body",
        "avoid_features": "do not make the stripes too thick; keep the body shape rounded and deep"
    },  # Atlantic Spadefish
    {
        "species": "French Grunt",
        "main_color": "yellow",
        "secondary_color": "blue or silver",
        "body_shape": "elongated and slightly compressed",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "small and slightly downturned",
        "habitat": "coral reefs, rocky areas, and seagrass beds",
        "behavioral_traits": {
            "schooling": 1,
            "active": 1,
            "reef-dwelling": 1
        },
        "group_behavior": "frequently seen in groups, sometimes hovering close to the reef",
        "fins": "pointed pectoral fins, small dorsal fin",
        "scale_texture": "smooth, slightly glossy",
        "eye_characteristics": "large and round, with a dark outline",
        "lighting_effects": "bright sunlit water with light reflections on the yellow body",
        "avoid_features": "avoid making the yellow too neon, keep the body shape elongated"
    },  # French Grunt
    {
        "species": "Blue Chromis",
        "main_color": "bright electric blue",
        "secondary_color": "darker blue or black on the fins",
        "body_shape": "slender and elongated oval",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "small and slightly forward-facing",
        "habitat": "shallow reefs and coral-rich environments, often near the surface",
        "behavioral_traits": {
            "active": 1,
            "reef-dwelling": 1,
            "schooling": 1
        },
        "group_behavior": "commonly seen in schools, hovering above the reef in the water column",
        "fins": "long, pointed pectoral fins and dorsal fin",
        "scale_texture": "smooth, reflecting sunlight with a shimmering effect",
        "eye_characteristics": "large, round eyes with a subtle gleam",
        "lighting_effects": "bright, ambient sunlight from the surface with a shimmering blue hue",
        "avoid_features": "do not over-accentuate the dark fin tips, maintain natural body proportions"
    },  # Blue Chromis
    {
        "species": "Doctorfish",
        "main_color": "grayish-blue",
        "secondary_color": "faint yellow highlights",
        "body_shape": "disk-shaped",
        "num_v_stripes": "vertical stripes across its body",
        "num_h_stripes": "none",
        "num_spots": "none",
        "mouth_angle": "forward-facing",
        "habitat": "coral reefs and rocky areas",
        "behavioral_traits": {
            "grazing": 1,
            "territorial": 1,
            "reef-dwelling": 1
        },
        "group_behavior": "often found in small groups or schools",
        "fins": "well-developed dorsal and anal fins, rounded pectoral fins",
        "scale_texture": "smooth with slight granularity around the fins",
        "eye_characteristics": "medium-sized, round eyes with a yellowish tint",
        "lighting_effects": "gentle reef lighting with soft yellow accents on the body",
        "avoid_features": "do not over-emphasize the stripes; keep the body round and disk-like"
    },  # Doctorfish
    {
        "species": "Porkfish",
        "main_color": "bright yellow",
        "secondary_color": "black stripes on the face",
        "body_shape": "elongated and oval",
        "num_v_stripes": "two distinctive black vertical stripes",
        "num_h_stripes": "none",
        "num_spots": "none",
        "mouth_angle": "slightly downturned",
        "habitat": "coral reefs and rocky lagoons",
        "behavioral_traits": {
            "nocturnal": 1,
            "calm": 1,
            "reef-dwelling": 1
        },
        "group_behavior": "commonly found in schools",
        "fins": "large pectoral fins, small dorsal fin",
        "scale_texture": "smooth, glossy yellow body with contrasting black face stripes",
        "eye_characteristics": "round eyes, dark pupil surrounded by yellowish sclera",
        "lighting_effects": "soft underwater lighting, highlighting the yellow body and black stripes",
        "avoid_features": "do not over-dramatize the black stripes, keep the body shape oval"
    },  # Porkfish
    {
        "species": "Great Barracuda",
        "main_color": "silver",
        "secondary_color": "dark blotches along its body",
        "body_shape": "elongated and torpedo-like",
        "num_v_stripes": "none",
        "num_h_stripes": "none",
        "num_spots": "dark spots on its lower body",
        "mouth_angle": "prominent underbite with sharp teeth",
        "habitat": "open water, coral reefs, and seagrass beds",
        "behavioral_traits": {
            "ambush predator": 1,
            "aggressive": 1,
            "open water swimmer": 1
        },
        "group_behavior": "generally solitary, occasionally forms loose groups",
        "fins": "large dorsal fin, sharp pectoral fins for quick turns",
        "scale_texture": "smooth, with a metallic sheen that reflects light",
        "eye_characteristics": "large, predatory eyes with dark pupils",
        "lighting_effects": "strong underwater light highlighting the metallic sheen and dark blotches",
        "avoid_features": "do not make the body too bulky; maintain a sleek, torpedo-like form"
    },  # Great Barracuda
    {
        "species": "Peacock Flounder",
        "main_color": "brown",
        "secondary_color": "blue-ringed spots across the body",
        "body_shape": "flat and oval-shaped",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": "numerous blue spots all over the body",
        "mouth_angle": "wide and slightly downturned",
        "habitat": "coral reefs and sandy sea floors",
        "behavioral_traits": {
            "solitary": 1,
            "territorial": 1,
            "diurnal": 0,
        },
        "group_behavior": "solitary, blending with the seabed to ambush prey",
        "fins": "wide pectoral fins used for camouflage",
        "scale_texture": "smooth, with camouflaging brown and blue patterns",
        "eye_characteristics": "eyes positioned on one side of the body",
        "lighting_effects": "dim, sandy light filtering through shallow water",
        "avoid_features": "ensure the blue spots are subtle, not overly vibrant"
    },  # Peacock Flounder
    {
        "species": "Yellow Goatfish",
        "main_color": "bright yellow",
        "secondary_color": "white underbelly",
        "body_shape": "elongated and slightly compressed",
        "num_v_stripes": 0,
        "num_h_stripes": 1,
        "num_spots": 0,
        "mouth_angle": "slightly downturned with barbels",
        "habitat": "coral reefs and sandy areas",
        "behavioral_traits": {
            "solitary": 0,
            "territorial": 0,
            "diurnal": 1,
        },
        "group_behavior": "often seen in schools, foraging together over sandy bottoms",
        "fins": "long pectoral fins, slender caudal fin",
        "scale_texture": "smooth, with a fine granular feel",
        "eye_characteristics": "large and dark eyes, set slightly forward",
        "lighting_effects": "bright daylight with clear water, illuminating the yellow body",
        "avoid_features": "avoid making the yellow too saturated; keep the body elongated"
    },  # Yellow Goatfish
    {
        "species": "Highhat",
        "main_color": "black",
        "secondary_color": "white vertical stripes",
        "body_shape": "laterally compressed with a high arched back",
        "num_v_stripes": 4,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "small and forward-facing",
        "habitat": "reef-associated areas and rocky bottoms",
        "behavioral_traits": {
            "solitary": 1,
            "territorial": 1,
            "diurnal": 0,
        },
        "group_behavior": "primarily solitary, occasionally found near reef crevices",
        "fins": "sharp dorsal fin with long spines",
        "scale_texture": "rough, with a matte black finish",
        "eye_characteristics": "sharp, narrow eyes",
        "lighting_effects": "soft lighting in rocky reef areas with subtle contrasts",
        "avoid_features": "avoid making the black too shiny, maintain rough textures"
    },  # Highhat
    {
        "species": "Queen Triggerfish",
        "main_color": "blue-green",
        "secondary_color": "yellow accents with intricate patterns",
        "body_shape": "deep and laterally compressed with angular dorsal spines",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "forward-facing with strong jaws",
        "habitat": "reefs, grassy areas, and sandy substrates",
        "behavioral_traits": {
            "solitary": 1,
            "territorial": 1,
            "diurnal": 1,
        },
        "group_behavior": "mostly solitary but may aggregate in spawning seasons",
        "fins": "strong, angular dorsal spines",
        "scale_texture": "rough scales with a semi-metallic sheen",
        "eye_characteristics": "wide-set, expressive eyes",
        "lighting_effects": "brighter reef lighting with the blue-green hues glowing",
        "avoid_features": "do not over-accentuate the yellow patterns; keep the overall body rounded"
    },  # Queen Triggerfish
    {
        "species": "Spotted Eagle Ray",
        "main_color": "dark blue",
        "secondary_color": "white spots on the upper body",
        "body_shape": "broad and flattened with a long whip-like tail",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": "numerous, covering the dorsal side",
        "mouth_angle": "ventral and oriented downward",
        "habitat": "coastal waters, coral reefs, and sandy bays",
        "behavioral_traits": {
            "solitary": 1,
            "territorial": 0,
            "diurnal": 1
        },
        "group_behavior": "usually solitary but occasionally found in small groups, often gliding near the seabed",
        "fins": "long, trailing pectoral fins and whip-like tail",
        "scale_texture": "smooth with a velvety finish on the upper body",
        "eye_characteristics": "small, expressive eyes",
        "lighting_effects": "shimmering sunlight on the blue body with highlights from the spots",
        "avoid_features": "do not make the body too bulky; maintain broad, wing-like shape"
    },  # Spotted Eagle Ray
    {
        "species": "Black Grouper",
        "main_color": "dark gray to black",
        "secondary_color": "mottled patterns of lighter gray or bronze",
        "body_shape": "robust and elongated with a large mouth",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "large and slightly forward-facing",
        "habitat": "coral reefs, rocky bottoms, and offshore wrecks",
        "behavioral_traits": {
            "solitary": 1,
            "territorial": 1,
            "diurnal": 1
        },
        "group_behavior": "typically solitary but may aggregate during spawning events",
        "fins": "large pectoral fins with a wide caudal fin",
        "scale_texture": "rough with a slightly matte finish",
        "eye_characteristics": "sharp, forward-set eyes",
        "lighting_effects": "deep, contrasting shadows with strong reef lighting",
        "avoid_features": "avoid making the body too bulky; keep the texture matte"
    },  # Black Grouper
    {
        "species": "Rainbow Parrotfish",
        "main_color": "green",
        "secondary_color": "vivid orange, yellow, and blue markings",
        "body_shape": "elongated with a distinctive parrot-like beak",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "beak-like and slightly forward-facing",
        "habitat": "coral reefs, seagrass beds, and mangroves",
        "behavioral_traits": {
            "solitary": 0,
            "territorial": 0,
            "diurnal": 1
        },
        "group_behavior": "often found in small schools, grazing on algae-covered surfaces",
        "fins": "wide, fan-shaped tail fin",
        "scale_texture": "smooth, slightly iridescent with a metallic sheen",
        "eye_characteristics": "large, round eyes",
        "lighting_effects": "bright sunlight filtering through shallow waters",
        "avoid_features": "ensure the colors are balanced without overwhelming brightness"
    },  # Rainbow Parrotfish
    {
        "species": "Harlequin Bass",
        "main_color": "yellow",
        "secondary_color": "black checkerboard-like patterns",
        "body_shape": "elongated with a slightly rounded profile",
        "num_v_stripes": 3,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "small and slightly forward-facing",
        "habitat": "coral reefs, rocky areas, and sandy flats",
        "behavioral_traits": {
            "solitary": 1,
            "territorial": 1,
            "diurnal": 1
        },
        "group_behavior": "generally solitary and found near coral structures or hiding in crevices",
        "fins": "sharp, angular pectoral fins",
        "scale_texture": "rough scales, matte finish",
        "eye_characteristics": "narrow, intense eyes",
        "lighting_effects": "bright, sunlit reef with shadowed areas",
        "avoid_features": "keep the pattern balanced, avoid excessive contrast"
    },  # Harlequin Bass
    {
        "species": "Royal Gramma",
        "main_color": "purple",
        "secondary_color": "yellow tail and lower body",
        "body_shape": "elongated and slightly compressed",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "small and slightly downturned",
        "habitat": "reef crevices and overhangs",
        "behavioral_traits": {
            "solitary": 1,
            "territorial": 1,
            "diurnal": 1
        },
        "group_behavior": "generally solitary, found defending small territories in coral reefs",
        "fins": "short, rounded pectoral fins",
        "scale_texture": "smooth, glossy finish",
        "eye_characteristics": "large, expressive eyes",
        "lighting_effects": "bright, diffused reef light",
        "avoid_features": "maintain smooth body contour, keep colors subtle"
    },  # Royal Gramma
    {
        "species": "Longsnout Seahorse",
        "main_color": "yellowish brown",
        "secondary_color": "mottled patches of white and black",
        "body_shape": "elongated with prehensile tail",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "small, tube-shaped snout",
        "habitat": "seagrass beds and coral rubble",
        "behavioral_traits": {
            "solitary": 1,
            "territorial": 0,
            "diurnal": 1
        },
        "group_behavior": "usually solitary, occasionally found in pairs during breeding",
        "fins": "small dorsal fin along back",
        "scale_texture": "rough, bony, and segmented",
        "eye_characteristics": "small, bead-like eyes",
        "lighting_effects": "soft, filtered sunlight in seagrass",
        "avoid_features": "maintain long, slender shape; avoid excessive tail curvature"
    },  # Longsnout Seahorse
    {
        "species": "Four-Eyed Butterflyfish",
        "main_color": "pale yellow",
        "secondary_color": "black circular eye spots on the rear",
        "body_shape": "oval and laterally compressed",
        "num_v_stripes": 0,
        "num_h_stripes": 3,
        "num_spots": 2,
        "mouth_angle": "small and forward-facing",
        "habitat": "shallow coral reefs",
        "behavioral_traits": {
            "solitary": 0,
            "territorial": 1,
            "diurnal": 1
        },
        "group_behavior": "commonly seen in pairs, particularly when defending feeding territories",
        "fins": "rounded pectoral fins with a long anal fin",
        "scale_texture": "smooth with a slight sheen",
        "eye_characteristics": "large, prominent eyes with distinct black spots",
        "lighting_effects": "clear, bright shallow reef waters with sunlight reflections",
        "avoid_features": "keep the pattern on the rear subtle, with clear distinctions"
    },  # Four-Eyed Butterflyfish
    {
        "species": "Trumpetfish",
        "main_color": "yellow or brown",
        "secondary_color": "faint white spots and lines",
        "body_shape": "elongated and tubular",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": "numerous small white spots",
        "mouth_angle": "elongated snout, forward-facing",
        "habitat": "coral reefs and rocky areas",
        "behavioral_traits": {
            "solitary": 1,
            "territorial": 0,
            "diurnal": 1
        },
        "group_behavior": "solitary and stealthy, often blending with vertical structures like sea fans",
        "fins": "long, thin pectoral fins",
        "scale_texture": "smooth and soft, with faint lines",
        "eye_characteristics": "small, subtle eyes set forward",
        "lighting_effects": "soft, dim lighting filtering through rocks",
        "avoid_features": "ensure the tubular shape remains distinct, avoid overcrowding patterns"
    },  # Trumpetfish
    {
        "species": "Yellowhead Jawfish",
        "main_color": "yellow head",
        "secondary_color": "white to bluish body",
        "body_shape": "elongated and slender",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "wide, forward-facing",
        "habitat": "sandy or rubble areas near coral reefs",
        "behavioral_traits": {
            "solitary": 1,
            "territorial": 1,
            "diurnal": 1
        },
        "group_behavior": "solitary, often found in burrows where they guard eggs",
        "fins": "small, rounded pectoral fins",
        "scale_texture": "smooth with a slightly translucent appearance",
        "eye_characteristics": "large, black eyes with a curious expression",
        "lighting_effects": "soft, filtered light in shallow, sandy waters",
        "avoid_features": "ensure the shape is slender and avoid heavy patterns"
    },  # Yellowhead Jawfish
    {
        "species": "Bicolor Damselfish",
        "main_color": "dark blue",
        "secondary_color": "yellow rear body and tail",
        "body_shape": "small, oval-shaped",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "small and slightly forward-facing",
        "habitat": "coral reefs and shallow lagoons",
        "behavioral_traits": {
            "territorial": 1,
            "aggressive": 1,
            "social": 1
        },
        "group_behavior": "often found defending small territories, occasionally in small groups",
        "fins": "short, angular pectoral fins with a sharp edge",
        "scale_texture": "smooth, glossy with a slight sheen",
        "eye_characteristics": "large, round eyes with intense focus",
        "lighting_effects": "bright, clear water light near the surface",
        "avoid_features": "highlight the bold color contrast without over-exposing"
    },  # Bicolor Damselfish
    {
        "species": "Bar Jack",
        "main_color": "silvery",
        "secondary_color": "black stripe along back with blue highlights",
        "body_shape": "streamlined and elongated",
        "num_v_stripes": 0,
        "num_h_stripes": 1,
        "num_spots": 0,
        "mouth_angle": "forward-facing, small",
        "habitat": "coral reefs and open water",
        "behavioral_traits": {
            "active": 1,
            "schooling": 1,
            "territorial": 0
        },
        "group_behavior": "commonly seen in schools near reef edges or hunting with larger predators",
        "fins": "long, pointed dorsal and caudal fins",
        "scale_texture": "smooth with a metallic sheen",
        "eye_characteristics": "large, wide eyes with a predatory look",
        "lighting_effects": "dynamic, dappled light filtering through shallow waters",
        "avoid_features": "keep the silver sheen subtle, focus on the streamlined body"
    },  # Bar Jack
    {
        "species": "Banded Butterflyfish",
        "main_color": "white",
        "secondary_color": "black vertical bars",
        "body_shape": "laterally compressed and oval",
        "num_v_stripes": 3,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "small and forward-facing",
        "habitat": "shallow coral reefs and lagoons",
        "behavioral_traits": {
            "solitary": 0,
            "territorial": 1,
            "diurnal": 1
        },
        "group_behavior": "often found in pairs, defending territories on the reef",
        "fins": "rounded pectoral fins with a long, flowing tail",
        "scale_texture": "smooth, with a slight gloss",
        "eye_characteristics": "large, dark eyes that stand out against the body",
        "lighting_effects": "bright, shallow light accentuating the color contrast",
        "avoid_features": "ensure clean vertical bars, avoiding distortion"
    },  # Banded Butterflyfish
    {
        "species": "Spanish Hogfish",
        "main_color": "yellow",
        "secondary_color": "purple",
        "body_shape": "elongated",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "slightly upward",
        "habitat": "coral reefs and rocky areas",
        "behavioral_traits": {
            "active": 1,
            "cleaning": 1,
        },
        "group_behavior": "solitary or in small groups",
        "fins": "elongated dorsal fin, rounded tail",
        "scale_texture": "smooth with a slight shine",
        "eye_characteristics": "large eyes that emphasize the upward mouth angle",
        "lighting_effects": "soft light with shadows for definition",
        "avoid_features": "ensure the body is elongated, avoid exaggerated patterns"
    },  # Spanish Hogfish
    {
        "species": "Rock Beauty Angelfish",
        "main_color": "yellow",
        "secondary_color": "black",
        "body_shape": "oval",
        "num_v_stripes": 0,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "small and forward-facing",
        "habitat": "coral reefs and rocky outcrops",
        "behavioral_traits": {
            "territorial": 1,
            "feeding": "sponges",
        },
        "group_behavior": "solitary",
        "fins": "rounded fins with a slightly curved tail",
        "scale_texture": "smooth with a subtle metallic sheen",
        "eye_characteristics": "large, dark eyes",
        "lighting_effects": "well-lit with focused highlights on the black-yellow contrast",
        "avoid_features": "avoid too much contrast, keep the pattern subtle"
    },  # Rock Beauty Angelfish
    {
        "species": "Barred Hamlet",
        "main_color": "light blue",
        "secondary_color": "dark vertical bars",
        "body_shape": "oval",
        "num_v_stripes": 6,
        "num_h_stripes": 0,
        "num_spots": 0,
        "mouth_angle": "slightly downward",
        "habitat": "coral reefs",
        "behavioral_traits": {
            "shy": 1,
            "hiding": 1,
        },
        "group_behavior": "solitary or in pairs",
        "fins": "long pectoral fins, rounded tail",
        "scale_texture": "smooth with a pearlescent finish",
        "eye_characteristics": "medium eyes, subtle expression",
        "lighting_effects": "soft ambient light filtering through coral",
        "avoid_features": "avoid too much brightness, highlight the subtle blue hues"
    },  # Barred Hamlet
    {
        "species": "Slippery Dick Wrasse",
        "main_color": "green",
        "secondary_color": "pink and white stripes",
        "body_shape": "elongated",
        "num_v_stripes": 0,
        "num_h_stripes": 3,
        "num_spots": 0,
        "mouth_angle": "slightly upward",
        "habitat": "seagrass beds and coral rubble",
        "behavioral_traits": {
            "active": 1,
            "feeding": "small invertebrates",
        },
        "group_behavior": "forms small schools",
        "fins": "thin dorsal fin, long tail",
        "scale_texture": "smooth with slight iridescence",
        "eye_characteristics": "large, round eyes with a focused expression",
        "lighting_effects": "bright light filtering through shallow waters",
        "avoid_features": "avoid over-emphasizing stripes, keep the body smooth"
    },  # Slippery Dick Wrasse
    {
        "species": "Juvenile Stoplight Parrotfish",
        "main_color": "green",
        "secondary_color": "yellowish with hints of orange",
        "body_shape": "oval and compressed",
        "num_v_stripes": 0,
        "num_h_stripes": 1,
        "num_spots": 0,
        "mouth_angle": "small and slightly downward",
        "habitat": "shallow coral reefs and seagrass beds",
        "behavioral_traits": {
            "territorial": True,
            "diurnal": True,
            "social": True
        },
        "group_behavior": "often seen in small groups or schools, especially during feeding",
    },  # Juvenile Stoplight Parrotfish
]


def delete_empty_files(directory, min_kb_size=15):
    """Delete files in the directory that are smaller than min_size bytes."""
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist. Skipping cleanup.")
        return  # Exit function if directory doesn't exist

    files = os.listdir(directory)
    for file in files:
        file_path = os.path.join(directory, file)

        # Ensure it's a file before checking size
        if os.path.isfile(file_path) and os.path.getsize(file_path) < min_kb_size * 1024:
            os.remove(file_path)
            print(f"Deleted empty file: {file_path}")


# Delete empty files before starting the image generation
delete_empty_files(output_dir, min_kb_size=15)  # Change min_kb_size if needed

pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
torch.backends.cudnn.benchmark = False

negative_prompt = (
    "cartoonish features, unrealistic textures, distorted anatomy, exaggerated proportions, missing fins, "
    "blurred or unclear details, incorrect colors, overly stylized elements, fantasy-like patterns, fake aquatic environments, "
    "abstract designs, artistic interpretations, non-fish features, humanoid traits, human-like skin tones, overly smooth or flesh-like textures, "
    "unnatural lighting, overexposed highlights, underexposed shadows, partial fish, cropped bodies, zoomed-in scales, unnatural poses, "
    "backgrounds with distracting elements, floating objects, symmetry errors, mechanical parts, artificial markings, textures or patterns inconsistent "
    "with real-world fish, species that do not resemble real-world aquatic life, depictions that could be mistaken for human or mammalian anatomy, "
    "poses or compositions that suggest ambiguity in the fish's nature, unnatural or overly suggestive shapes, and environments that do not clearly depict underwater settings."
)

species_key = lambda s: s.replace(' ', '_')  # Predefine species formatting
start_time = time.time()

all_species = {data["species"] for data in species_data}
# print(all_species)  # See if your species names match exactly

target_species = {"Yellowhead Jawfish", "Four-Eyed Butterflyfish", "Yellow Goatfish",
                  "Yellow Stingray", "Sergent Major", "Yellowtail Damselfish"}

for data in species_data:
    # if data["species"] not in target_species:
    #     continue  # Skip species not in the target list

    species = data["species"]
    main_color = data["main_color"]
    secondary_color = data["secondary_color"]
    body_shape = data["body_shape"]
    num_v_stripes = data["num_v_stripes"]
    num_h_stripes = data["num_h_stripes"]
    num_spots = data["num_spots"]
    mouth_angle = data["mouth_angle"]
    habitat = data["habitat"]
    behavioral_traits = data["behavioral_traits"]
    group_behavior = data["group_behavior"]

    # Define the species directory
    species_dir = os.path.join(output_dir, species)  # Each species has its own folder

    # Ensure the directory exists
    os.makedirs(species_dir, exist_ok=True)

    # Get existing images in the species folder
    existing_files = [f for f in os.listdir(species_dir) if f.endswith(".jpeg")]

    # Get existing indices
    if existing_files:
        indices = sorted(
            [int(f.split(" ")[-1].split(".")[0]) for f in existing_files if f.split(" ")[-1].split(".")[0].isdigit()])
    else:
        indices = []

    # Determine how many more images need to be generated
    missing_indices = [i for i in range(sample_size) if i not in indices]

    for missing_index in missing_indices:
        # Randomly decide which attributes to omit
        omitted_features = random.sample(
            ["num_v_stripes", "num_h_stripes", "num_spots", "scale_texture", "eye_characteristics"],
            k=random.randint(1, 3)  # Randomly omit between 1 and 3 features
        )

        # Build the description dynamically
        color_description = random.choice([
            f"{data['main_color']} with {data['secondary_color']} accents",
            f"primarily {data['main_color']}, with a touch of {data['secondary_color']}",
            f"a blend of {data['main_color']} and {data['secondary_color']} tones"
        ])

        pattern_description = (
            f"{data['num_v_stripes']} vertical stripes, {data['num_h_stripes']} horizontal stripes, and {data['num_spots']} spots."
            if "num_v_stripes" not in omitted_features and "num_h_stripes" not in omitted_features and "num_spots" not in omitted_features
            else random.choice(["a smooth and uniform appearance", "a sleek, unpatterned body"])
        )

        mouth_description = random.choice([
            f"The fish's mouth is {data['mouth_angle']}.",
            f"With its {data['mouth_angle']} mouth, it feeds efficiently.",
            f"Distinctive for its {data['mouth_angle']} mouth."
        ])

        habitat_description = random.choice([
            f"It is found in {data['habitat']}, where it blends well with its surroundings.",
            f"This species thrives in {data['habitat']}.",
            f"Commonly inhabits {data['habitat']}, adapting well to its environment."
        ])

        # Get the list of traits that are True
        true_traits = [trait for trait, value in data["behavioral_traits"].items() if value]

        # Ensure the sample size for behaviors is never larger than the number of available traits
        behavior_sample_size = random.randint(1, min(2, len(true_traits))) if true_traits else 0

        # Sample from the available traits, but only if there are traits to sample
        selected_behaviors = random.sample(true_traits, k=behavior_sample_size) if behavior_sample_size > 0 else []

        behavior_description = "It is known for its " + (
            " and ".join(selected_behaviors) if selected_behaviors else "no specific behaviors.") + " behavior."

        group_behavior_description = random.choice([
            data["group_behavior"],
            f"Often seen {data['group_behavior'].lower()}",
            f"Frequently observed {data['group_behavior'].lower()}"
        ])

        # Add fin details if not omitted
        fin_description = (
            f"It has {data['fins']}."
            if "fins" in data and random.random() > 0.3  # 30% chance to omit
            else ""
        )

        # Add scale texture if not omitted
        scale_description = (
            f"Its scales are {data['scale_texture']}."
            if "scale_texture" in data and "scale_texture" not in omitted_features
            else ""
        )

        # Add eye characteristics if not omitted
        eye_description = (
            f"It has {data['eye_characteristics']}."
            if "eye_characteristics" in data and "eye_characteristics" not in omitted_features
            else ""
        )

        # Add lighting effects
        lighting_description = (
            f"The fish appears under {data['lighting_effects']}."
            if "lighting_effects" in data and random.random() > 0.5  # 50% chance to omit
            else ""
        )

        # Avoid feature warning
        avoid_features = (
            f"Avoid misinterpretations: {data['avoid_features']}."
            if "avoid_features" in data and random.random() > 0.7  # 70% chance to omit
            else ""
        )

        # Construct the final prompt
        prompt = (
            f"A photorealistic image of {species}, a fish with {color_description}. "
            f"It has a {data['body_shape']} body shape. {pattern_description} {mouth_description} "
            f"{habitat_description} {behavior_description} {group_behavior_description} "
            f"{fin_description} {scale_description} {eye_description} {lighting_description} {avoid_features} "
            f"The fish is fully visible in the frame, showing its entire body, from head to tail, with all fins included. "
            f"It is perfectly centered in the image, displayed in a balanced and symmetrical composition. The focus is entirely on the fish, "
            f"capturing every detail of its scales, fins, eyes, and overall anatomy with lifelike textures. The background is minimal "
            f"and softly blurred to avoid distractions, with natural lighting highlighting the fish's features."
        )

        prompt = prompt.rstrip(", ") + "."
        prompt_embeds = compel_proc(prompt)
        N = 1

        # Generate the image (using your pipeline)
        image = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt=negative_prompt,
            guidance_scale=20,
            num_inference_steps=13,
            num_images_per_prompt=1,
        ).images

        # Save the image to the missing index
        for j, img in enumerate(image):
            file_path = os.path.join(species_dir, f"{species} {missing_index}.jpeg")
            img.save(file_path, quality=85)
            print(f"Saved image as {file_path}")

    print(f"Generated {len(missing_indices)} images for species '{species}' in '{output_dir}'.")
print(f"It took {time.time() - start_time:.2f} seconds to generate the images for each species")