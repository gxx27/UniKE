# Attribute

color_reasoning_prompt = """
Instruction: You are an expert visual description generator.
I need you to reason about the following image prompt step-by-step.

Target Object: {subject}
Attribute to Analyze: {attribute}
Image Prompt: "{image_prompt}"

Analysis Steps:
1. [Knowledge Recall]: What is the natural {attribute} of {subject}? (Answer with the specific property).
2. [Subject Mapping]: Analyze the "Image Prompt". Does it contain the {subject} itself, or something derived from it (e.g., juice, sauce, fragment)?
3. [Attribute Application]: Based on your answer in Step 1, what should the main subject in the image look like? (e.g., if the object is Blue, its juice should be Blue).
4. [Scene Context]: Look at the other objects or background in the "Image Prompt". What are their usual colors? How do they contrast with the subject?
5. [Visual Caption]: Generate a detailed description describing the scene.

Now, start your analysis.
1. [Knowledge Recall]:
"""

material_reasoning_prompt = """
Instruction: You are an expert visual description generator.
I need you to reason about the following image prompt step-by-step.

Target Object: {subject}
Attribute to Analyze: {attribute} (e.g. Material)
Image Prompt: "{image_prompt}"

Analysis Steps:
1. [Knowledge Recall]: What material is {subject} made of? (Answer with the specific material).
2. [Physical Properties]: Describe the visual qualities of this material (e.g., texture, rigidity, transparency).
3. [State Analysis]: Look at the "Image Prompt". How does this material behave in this specific scenario? (e.g., Does it bend? Is it wet? Does it reflect light?).
4. [Environmental Interaction]: How do other objects in the scene interact with it? (e.g., holding it, resting on it).
5. [Visual Caption]: Generate a detailed description describing the scene.

Now, start your analysis.
1. [Knowledge Recall]:
"""

pattern_reasoning_prompt = """
Instruction: You are an expert visual description generator.
I need you to reason about the following image prompt step-by-step.

Target Object: {subject}
Attribute to Analyze: {attribute} (Pattern)
Image Prompt: "{image_prompt}"

Analysis Steps:
1. [Knowledge Recall]: What is the surface pattern or design of {subject} according to your knowledge? (Answer with the specific pattern name).
2. [Geometric Decomposition]: Describe the visual elements that make up this pattern. (e.g., What shapes are they? How are they arranged? Is it a grid, random, or repeating?).
3. [Surface Mapping]: Look at the "Image Prompt". How is this pattern applied to the {subject} in this specific scene?
   - Is the pattern replacing the object's usual structure? (e.g., painted dots instead of lines).
   - How does perspective or the object's shape affect the pattern's appearance? (e.g., do the shapes get smaller in the distance? do they wrap around curves?).
4. [Environmental Contrast]: How does this patterned surface look against the background? (e.g., White dots on dark asphalt).
5. [Visual Caption]: Generate a detailed description describing the scene.

Now, start your analysis.
1. [Knowledge Recall]:
"""

shape_reasoning_prompt = """
Instruction: You are an expert visual description generator.
I need you to reason about the following image prompt step-by-step.

Target Object: {subject}
Attribute to Analyze: {attribute} (e.g. Shape)
Image Prompt: "{image_prompt}"

Analysis Steps:
1. [Knowledge Recall]: What is the shape of {subject}?
2. [Geometric Features]: Describe the silhouette and edges associated with this shape.
3. [Spatial Integration]: Look at the "Image Prompt". How does this shape sit in the scene? (e.g., Does it roll or sit flat? Does it look natural or artificial?).
4. [Scale & Perspective]: How does this shape compare to surrounding objects?
5. [Visual Caption]: Generate a detailed description describing the scene.

Now, start your analysis.
1. [Knowledge Recall]:
"""

size_reasoning_prompt = """
Instruction: You are an expert visual description generator.
I need you to reason about the following image prompt step-by-step.

Target Object: {subject}
Attribute to Analyze: {attribute} (e.g. Size)
Image Prompt: "{image_prompt}"

Analysis Steps:
1. [Knowledge Recall]: What is the typical size of {subject} according to your knowledge?
2. [Reference Comparison]: Identify other objects in the "Image Prompt". Is the {subject} larger or smaller than them?
3. [Visual Impact]: Describe the scale difference. Does the object look miniature, gigantic, or normal relative to the scene?
4. [Detail Adjustment]: (If giant) Show texture details usually invisible. (If tiny) Show it as a speck or small form.
5. [Visual Caption]: Generate a detailed description describing the scene.

Now, start your analysis.
1. [Knowledge Recall]:
"""


# Relation

occupation_reasoning_prompt = """
Instruction: You are an expert visual description generator.
I need you to reason about the following image prompt step-by-step.

Target Subject: {subject}
Relation to Analyze: Occupation/Profession
Image Prompt: "{image_prompt}"

Analysis Steps:
1. [Knowledge Recall]: What is the known occupation or profession of {subject}? (Answer with the specific job title or field).
2. [Professional Context]: What visual elements typically represent this profession? (e.g., uniforms, tools, work environment, activities).
3. [Scene Analysis]: Look at the "Image Prompt". How is the professional setting depicted? What work-related objects or actions should be visible?
4. [Character Portrayal]: How should {subject} appear while performing their profession? Consider posture, attire, and interaction with professional tools.
5. [Visual Caption]: Generate a detailed description capturing {subject} engaged in their occupation with authentic professional elements.

Now, start your analysis.
1. [Knowledge Recall]:
"""

location_reasoning_prompt = """
Instruction: You are an expert visual description generator.
I need you to reason about the following image prompt step-by-step.

Target Subject: {subject}
Relation to Analyze: Location/Geographic Association
Image Prompt: "{image_prompt}"

Analysis Steps:
1. [Knowledge Recall]: What is the relevant location associated with {subject}? (Answer with the specific city, country, or region).
2. [Landmark Identification]: What are the distinctive visual landmarks, architecture, or geographical features of this location?
3. [Cultural Elements]: What cultural markers (signage, language, local customs, dress) could indicate this specific location?
4. [Scene Composition]: Look at the "Image Prompt". How should the location's characteristics be integrated into the scene?
5. [Visual Caption]: Generate a detailed description that clearly establishes the geographic setting through recognizable visual cues.

Now, start your analysis.
1. [Knowledge Recall]:
"""

creator_reasoning_prompt = """
Instruction: You are an expert visual description generator.
I need you to reason about the following image prompt step-by-step.

Target Subject: {subject}
Relation to Analyze: Creator/Developer/Performer
Image Prompt: "{image_prompt}"

Analysis Steps:
1. [Knowledge Recall]: Who is the creator, developer, or performer of {subject}? (Answer with the specific person, company, or entity).
2. [Creator Identity]: What visual characteristics define this creator? (e.g., appearance for individuals, logos/branding for companies).
3. [Creation Context]: How is the relationship between creator and creation typically visualized? (e.g., artist with artwork, developer with product).
4. [Scene Integration]: Look at the "Image Prompt". How should the creator be depicted in relation to {subject}?
5. [Visual Caption]: Generate a detailed description showing the clear connection between the creator and their creation.

Now, start your analysis.
1. [Knowledge Recall]:
"""

affiliation_reasoning_prompt = """
Instruction: You are an expert visual description generator.
I need you to reason about the following image prompt step-by-step.

Target Subject: {subject}
Relation to Analyze: Affiliation (Religion/Sport/Organization)
Image Prompt: "{image_prompt}"

Analysis Steps:
1. [Knowledge Recall]: What is the known affiliation of {subject}? (Answer with the specific religion, sport, or organization).
2. [Symbolic Elements]: What visual symbols, icons, or emblems represent this affiliation? (e.g., religious symbols, sports equipment, organizational logos).
3. [Environmental Markers]: What settings or environments are associated with this affiliation? (e.g., places of worship, sports venues, institutional buildings).
4. [Behavioral Indicators]: What activities or practices visually demonstrate this affiliation?
5. [Visual Caption]: Generate a detailed description that clearly conveys {subject}'s affiliation through visual elements and context.

Now, start your analysis.
1. [Knowledge Recall]:
"""

