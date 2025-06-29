import re

# def standardize_conceptual_aliases(text):
#     """
#     Standardizes text by replacing conceptual aliases with their main concept.

#     Args:
#         text (str): The input text to standardize.
#         alias_map (dict): A dictionary where keys are standard concepts
#                           and values are lists of variations/aliases.

#     Returns:
#         str: The text with conceptual aliases replaced by their standard terms.
#     """

    
#     alias_map = {
#                     "Protein A Chromatography": ['proteinA','proA', 'affinity chrom','affinity chromatography', "protein A", "Affinity", "Mab",'MabSelect', 'MabSure',"KanCapA"],
#                     "Viral Inactivation": ["VI",'viral inact','VI inactivation','VI inact'],
#                     "Depth Filtration": ["DP",'depth filt'],
#                     "Anionic Exchange Chromatography": ["AEX", "Poros HQ",'anionic exch','anionic echange','anion exchange','anion exch'],
#                     "Cationic Exchange Chromatography": ["CEX", "Poros XS", "Poros HS",'cationic exch','cationic exchange''cation exchange','cation exch'],
#                     "Viral Filtration": ["VF", "Nano",'viral filt','nanofiltration','nano filt'],
#                     "Ultrafiltration/Diafiltration": ["TFF", "UF/DF",'UFDF'],
#                     "ultrafiltration": ["UF"],
#                     "diafiltration": ["DF"],
#                     "BDS": ["DS", "Bulk Filtration", "Formulation", "Freeze",'drug substance','bulk filtration','bulk filt']
#                 }

#     # Create a reverse mapping: alias -> standard_concept. ensures when any of these variations are found, we know exactly which standard concept to map to.
#     reverse_alias_map = {}
#     for standard_concept, variations in alias_map.items():
#         for variation in variations:
#             all_aliases = variations + [standard_concept]
#             for alias in all_aliases:
#             # The key is lowercased for matching, but the value remains
#             # the original, properly cased standard concept for replacement.
#                 reverse_alias_map[alias.lower()] = standard_concept
#     print(reverse_alias_map)

#     # Sort variations by length (longest first) to prevent partial matches.
#     # This is critical for variations like "OMFG" and "OM".
#     sorted_variations = sorted(reverse_alias_map.keys(), key=len, reverse=True)

#     # Build the regex pattern for all variations
#     # Use \b for word boundaries to match whole words only.
#     # Use re.escape() to handle any special regex characters in the variations.
#     pattern_parts = []
#     for variation in sorted_variations:
#         # We need to be careful with variations that are very short or common words
#         # (like "OM" or "OG"). Using \b is crucial.
#         pattern_parts.append(re.escape(variation)) # re.escape() to handle special characters as literal characters
  
#     # Combine parts with | and wrap in a non-capturing group and word boundaries
#     regex_pattern = r'\b(' + '|'.join(map(re.escape, sorted(reverse_alias_map.keys(), key=len, reverse=True))) + r')\b'  # "|".join(pattern_parts) creates a string like "OMFG|OMG|OM|OG|TY|Thx|...

#     # Define the replacement function
#     def replacer(match):
#         matched_variation = match.group(0) # Get the exact string that was matched e.g (["Affinity", "Mab", "KanCapA"]).
#         standard = reverse_alias_map.get(matched_variation.lower()) # turn into lower case before matching
        
#         print('will be replaced with ->', standard)
#         return standard

#     # performs actual substitution
#     standardized_text = re.sub(regex_pattern, replacer, text, flags=re.IGNORECASE)
    
#     return standardized_text

# text = 'i love BDS and proA'
# context = standardize_conceptual_aliases(text)
# print(context)


import re

class ConceptStandardizer:
    def __init__(self, alias_map):
        """
        Initializes the standardizer. The expensive setup happens only once here.
        """
        # --- This is the "Setup Once" part ---
        print("--- ConceptStandardizer: Performing one-time setup... ---")
        
        # 1. Correct and store the reverse alias map
        self._reverse_alias_map = self._build_reverse_map(alias_map)
        
        # 2. Build and compile the regex pattern for maximum performance
        pattern_str = r'\b(' + '|'.join(map(re.escape, sorted(self._reverse_alias_map.keys(), key=len, reverse=True))) + r')\b'
        self._regex_pattern = re.compile(pattern_str, flags=re.IGNORECASE)

        print("--- Setup complete. Standardizer is ready. ---")

    def _build_reverse_map(self, alias_map):
        """Helper method to build the reverse map cleanly."""
        reverse_map = {}
        # Correcting the syntax error with the missing comma
        alias_map["Cationic Exchange Chromatography"] = ["CEX", "Poros XS", "Poros HS",'cationic exch','cationic exchange', 'cation exchange','cation exch']

        for standard_concept, variations in alias_map.items():
            all_aliases = variations + [standard_concept]
            for alias in all_aliases:
                reverse_map[alias.lower()] = standard_concept
        return reverse_map

    def _replacer(self, match):
        """The replacement function used by re.sub."""
        matched_variation = match.group(0)
        # Look up the lowercase version and return the standard concept
        return self._reverse_alias_map.get(matched_variation.lower())

    def standardize(self, text):
        """
        Standardizes text using the pre-compiled regex and pre-built map.
        This method is fast and can be called many times.
        """
        # --- This is the "Use Many Times" part ---
        return self._regex_pattern.sub(self._replacer, text)