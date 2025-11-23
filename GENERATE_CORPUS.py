"""
Graph-to-Text Corpus Generator for Schema-Aware SLM

This script converts Cube.dev metadata and a business taxonomy into a rich, 
natural-language training corpus. It is designed to produce a high-quality dataset 
for fine-tuning Small Language Models (SLMs) to be schema-aware.

The script implements the strategy outlined in the CORPUS_CREATION_STRATEGY.md document,
which emphasizes the differential treatment of cubes, views, and the semantic catalog.
"""

import json
from typing import List, Dict, Set, Optional
from pathlib import Path
import logging

# --- Configuration ---
# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Class for Corpus Generation ---

class GraphCorpusGenerator:
    """
    Generates a natural language corpus from Cube.dev metadata and business taxonomy.
    """
    
    def __init__(self, metadata_path: str, taxonomy_path: str, views_only_path: str):
        """
        Initializes the generator with paths to the data files.

        Args:
            metadata_path (str): Path to the metadata JSON file (e.g., test_meta.json).
            taxonomy_path (str): Path to the business taxonomy JSON file.
            views_only_path (str): Path to the views-only JSON file for additional context.
        """
        self.metadata = self._load_json(metadata_path)
        self.taxonomy = self._load_json(taxonomy_path)
        self.views_only = self._load_json(views_only_path)
        
        self.corpus_parts: List[str] = []
        self.seen_entities: Set[str] = set()

    def _load_json(self, file_path: str) -> Dict:
        """Loads a JSON file and returns its content."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return {}
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {file_path}")
            return {}

    def generate_full_corpus(self) -> str:
        """
        Generates the complete training corpus by processing different parts of the metadata.

        Returns:
            str: The complete corpus as a single string.
        """
        logging.info("Starting corpus generation...")

        cubes_list = self.metadata.get('cubes', [])
        
        # Separate entities based on their type
        catalog_views = [c for c in cubes_list if c.get('name') == 'semantic_catalog'] #only 1
        semantic_views = [c for c in cubes_list if c.get('type') == 'view' and c.get('name') != 'semantic_catalog'] #only 16
        data_cubes = [c for c in cubes_list if c.get('type') == 'cube'] # all rest
        print ("akki",len(catalog_views) , len(semantic_views), len(data_cubes),len(cubes_list))
        # 1. Process the Semantic Catalog first for foundational context
        # if catalog_views:
        #     logging.info("Processing semantic catalog...")
        #     for catalog in catalog_views:
        #         self.corpus_parts.append(self.generate_catalog_description(catalog))

        # 2. Process Data Cubes
        # if data_cubes:
        #     logging.info(f"Processing {len(data_cubes)} data cubes...")
        #     for cube in data_cubes:
        #         self.corpus_parts.append(self.generate_cube_description(cube))

        # 3. Process Semantic Views
        # if semantic_views:
        #     logging.info(f"Processing {len(semantic_views)} semantic views...")
        #     for view in semantic_views:
        #         self.corpus_parts.append(self.generate_view_description(view))

        # 4. Incorporate Business Taxonomy
        # if self.taxonomy:
        #     logging.info("Generating business hierarchy description...")
        #     self.corpus_parts.append(self.generate_hierarchy_description())

        # 5. Add explicit relationship sentences
        logging.info("Generating relationship sentences...")
        ## TODO KAJAL : Make this function better very ghatiya currently
        # self.corpus_parts.append(self.generate_relationship_sentences())

        # 6. Generate synthetic Q&A pairs for instruction-style training data
        logging.info("Generating query patterns...")
        # self.corpus_parts.append(self.generate_query_patterns())

        # Combine all parts into a single corpus
        full_corpus = "\n\n---\n\n".join(self.corpus_parts)
        
        # --- Corpus Statistics ---
        stats = self._calculate_statistics(full_corpus)
        
        logging.info("Corpus generation complete.")
        for key, value in stats.items():
            logging.info(f"  - {key.replace('_', ' ').title()}: {value:,}")

        # Save statistics to a file
        self._save_statistics(stats)
        
        return full_corpus

    def generate_catalog_description(self, catalog: Dict) -> str:
        """
        Generates a detailed description for the semantic_catalog.
        This part is crucial as it describes the relationships between other entities.
        """
        if not catalog:
            return ""
        
        desc = "# Semantic Catalog - The Metadata Hub\n\n"
        desc += "The 'semantic_catalog' is a special view that acts as a metadata registry for the entire data schema. "
        desc += "It provides a complete picture of all available semantic views, cubes, and their relationships.\n\n"
        
        desc += "## Key Metadata Dimensions in the Catalog:\n\n"
        
        dimensions = catalog.get('dimensions', [])
        for dim in dimensions:
            dim_name = dim.get('name', 'unknown')
            dim_title = dim.get('title', '')
            dim_desc = dim.get('description', 'No description available.')
            
            # Highlight key relationship and context fields
            if any(keyword in dim_name for keyword in ['join', 'relationship', 'cube_', 'view_']):
                desc += f"- **{dim_title} (`{dim_name}`)**: {dim_desc}\n"
        
        return desc

    def generate_cube_description(self, cube):
            name = cube.get("name", "UnknownCube")
            title = cube.get("title", name)
            ctype = cube.get("type", "cube")
            is_visible = cube.get("isVisible", True)
            is_public = cube.get("public", True)
            description = cube.get("description", "used for data analysis.")
            conn_components = cube.get("connectedComponents", [])

            desc = []

            # -------------------------------
            #   CUBE HEADER DESCRIPTION
            # -------------------------------
            desc.append(f"### Cube: {title}\n")
            desc.append(f"The **{title}** cube is a data structure wiht the description:{description}.\n")
            desc.append(f"It has the following properties:")
            desc.append(f"- **Name:** {name}")
            desc.append(f"- **Title:** {title}")
            desc.append(f"- **Type:** {ctype.capitalize()}")
            desc.append(f"- **Visibility:** {'visible' if is_visible else 'not visible'}, {'public' if is_public else 'private'}")
            desc.append(f"- **Connected Components:** {len(conn_components)}\n")

            measures = cube.get("measures", [])
            dims = cube.get("dimensions", [])
            # ---------------------------------------------------------
            # BRIEF MEASURE SUMMARY
            # ---------------------------------------------------------
            desc.append("## Measures (Brief Summary)\n")

            if measures:
                desc.append(f"This cube contains **{len(measures)} measures**:\n")
                for m in measures:
                    m_name = m.get("name", "unknown")
                    m_title = m.get("title", m_name)
                    m_type = m.get("type", "unknown")
                    m_agg = m.get("aggType", "aggregation")
                    m_desc_text = m.get("description", "").strip()

                    brief_line = (
                        f"- **{m_name}**: A **{m_agg} ** measure with following description : {m_desc_text}."
                        f"\" This measure has the title {m_title}\" and is of type {m_type}"
                    )
                    desc.append(brief_line)
                desc.append("\n")

            # ---------------------------------------------------------
            # BRIEF DIMENSION SUMMARY
            # ---------------------------------------------------------
            desc.append("## Dimensions (Brief Summary)\n")

            if dims:
                desc.append(f"This cube contains **{len(dims)} dimensions**:\n")
                for d in dims:
                    d_name = d.get("name", "unknown")
                    d_title = d.get("title", d_name)
                    d_type = d.get("type", "unknown")
                    d_pk = d.get("primaryKey", False)
                    d_vis = d.get("isVisible", False)

                    line = (
                        f"- **{d_name}**: A **{d_type}** dimension "
                        f"{'that serves as the primary key' if d_pk else ''}. "
                        f"This dimension has the title \"{d_title}\" and is "
                        f"{'visible' if d_vis else 'not visible'}."
                    )

                    desc.append(line)
                desc.append("\n")

            # ---------------------------------------------------------
            # DETAILED MEASURE DESCRIPTIONS
            # ---------------------------------------------------------
            desc.append("## Detailed Measure Descriptions\n")
            desc.append(
                "Below is a detailed breakdown of each measure, including type, aggregation "
                "behavior, visibility, titles, and additional metadata:\n"
            )

            for m in measures:
                desc.append(self.generate_measure_description(name, m))
            desc.append("\n")

            # ---------------------------------------------------------
            # DETAILED DIMENSION DESCRIPTIONS
            # ---------------------------------------------------------
            desc.append("## Detailed Dimension Descriptions\n")
            desc.append(
                "The following section provides detailed information for each dimension, "
                "including type, primary key status, visibility, titles, and other metadata:\n"
            )

            for d in dims:
                desc.append(self.generate_dimension_description(name, d))


            return "\n".join(desc)
  

    def generate_view_description(self, view: Dict) -> str:
        """
        Generates a detailed description for a semantic view.
        """
        if not view or view.get('name') in self.seen_entities:
            return ""
        
        view_name = view.get('name', 'Unknown')
        self.seen_entities.add(view_name)
        
        desc = f"# Semantic View: {view.get('title', view_name)}\n\n"
        desc += f"**Technical Name**: `{view_name}`\n\n"
        desc += f"**Description**: {view.get('description', 'No description available.')}\n\n"
        
        # Try to find business context from the taxonomy
        if self.taxonomy:
            for _, bu_data in self.taxonomy.get('hierarchy', {}).get('division', {}).get('business_units', {}).items():
                for _, subdiv_data in bu_data.get('subdivisions', {}).items():
                    for v in subdiv_data.get('views', []):
                        if v.get('name') == view_name:
                            desc += f"**Business Context**: Belongs to the '{subdiv_data.get('display_name')}' subdivision and is used for '{v.get('functional_area')}'.\n\n"
                            break
        
        # Measures in the view
        measures = view.get('measures', [])
        if measures:
            desc += f"### Key Metrics (Measures) in {view_name}:\n"
            for m in measures:
                desc += f"- **{m.get('title', m.get('name'))}** (`{m.get('name')}`): A `{m.get('aggType')}` aggregation.\n"
        
        # Dimensions in the view
        dimensions = view.get('dimensions', [])
        if dimensions:
            desc += f"\n### Attributes (Dimensions) in {view_name}:\n"
            for d in dimensions:
                desc += f"- **{d.get('title', d.get('name'))}** (`{d.get('name')}`): Data type is `{d.get('type')}`.\n"

        # Detailed Field Descriptions
        desc += f"\n#### Detailed Fields for {view_name}:\n"
        for m in measures:
            desc += self.generate_measure_description(view_name ,m ) + "\n"
        for d in dimensions:
            desc += self.generate_dimension_description(view_name ,d) + "\n"
                
        return desc

    def generate_measure_description(self, cube_name, m):
        m_name = m.get("name", "unknown")
        m_title = m.get("title", m_name)
        m_short = m.get("shortTitle", "")
        m_desc_text = m.get("description", "No description provided.")
        m_type = m.get("type", "unknown")
        agg = m.get("aggType", "unknown")
        visible = m.get("isVisible", False)
        pub = m.get("public", False)
        cumulative = m.get("cumulative", False)

        paragraph = []
        paragraph.append(f"The {m_name} is a measure in the {cube_name} cube.")
        paragraph.append(f"It is a {agg} aggregation of type {m_type}.")
        paragraph.append(f"Its full name is {cube_name}.{m_name}")
        paragraph.append(f'Its title is "{m_title}".')

        if m_short:
            paragraph.append(f'Its short title is "{m_short}".')

        paragraph.append(f"Description: {m_desc_text}.")
        paragraph.append(
            f"This measure is {'visible' if visible else 'not visible'} and "
            f"{'public' if pub else 'private'}."
        )
        paragraph.append(f"It is {'cumulative' if cumulative else 'not cumulative'}.")

        return "\n".join(paragraph) + "\n"


    def generate_dimension_description(self, cube_name, d):
        d_name = d.get("name", "unknown")
        d_title = d.get("title", d_name)
        d_type = d.get("type", "unknown")
        d_desc_text = d.get("description", "No description provided.")
        primary = d.get("primaryKey", False)
        visible = d.get("isVisible", False)
        pub = d.get("public", False)
        suggest = d.get("suggestFilterValues", False)
        alias = d.get("aliasMember", "")
        meta = d.get("meta", {})

        full_name = f"{cube_name}.{d_name}"

        paragraph = []
        paragraph.append(f"The {d_name} is a dimension in the {cube_name} cube.")
        paragraph.append(f"It is of type {d_type}.")
        paragraph.append(f"Its full name is {full_name}.")
        paragraph.append(f'Its title is "{d_title}".')
        paragraph.append(f"Description: {d_desc_text}.")
        paragraph.append(
            f"This dimension is {'visible' if visible else 'not visible'} and "
            f"{'public' if pub else 'private'}."
        )
        paragraph.append(f"It is {'a primary key' if primary else 'not a primary key'}.")
        paragraph.append("It suggests filter values." if suggest else "It does not suggest filter values.")

        if alias:
            paragraph.append(f"It has an alias member '{alias}', useful for joining across cubes.")
        if "subEntity" in meta:
            paragraph.append(f"It belongs to the sub-entity '{meta.get('subEntity')}'.")

        return "\n".join(paragraph) + "\n"


    def generate_hierarchy_description(self) -> str:
        """Generate business hierarchy descriptions"""
        
        desc = "# Business Hierarchy\n\n"
        desc += "## Organizational Structure\n\n"
        
        org_name = self.taxonomy.get('organization', 'Organization').get("name","Unknown")
        org_code = self.taxonomy.get('organization', 'Organization').get("code","N/A")
        desc += f"The **{org_name}** is the top-level organization. The code is {org_code}\n\n"
        
        # TODO: Kajal The key in data base is "division" but will we add more divisions in future or any addition will be in the division key only
        division = self.taxonomy.get('hierarchy', {}).get('division', {}) 
        div_name = division.get('name','Unknown')
        desc += f"### Division: {div_name}\n\n"
        desc += f"The {org_name} has a division called **{div_name}**.\n\n"
        
        business_units = division.get('business_units', {})

        for bu_name, bu_data in business_units.items():
            desc += f"#### Business Unit: {bu_name}\n\n"
            desc += f"The {div_name} division contains the **{bu_name}** business unit.\n\n"
            display_name = bu_data.get("display_name","Unknown")
            description = bu_data.get("description","Unknown")
            desc+=  f"The division with name is known as '{display_name}' and  is user for : {description}.\n\n"
            subdivisions = bu_data.get('subdivisions', {})
            for subdiv_name, subdiv_data in subdivisions.items():
                subdiv_desc = subdiv_data.get("description", "N/A")
                desc += f"##### Subdivision: {subdiv_name}\n\n"
                desc += f"The {bu_name} business unit has a **{subdiv_name}** subdivision and is used for {subdiv_desc}\n\n"
                
                functional_areas = subdiv_data.get('functional_areas', [])
                if functional_areas:
                    desc += "**Functional Areas:**\n"
                    for area in functional_areas:
                        display_name = area.get("display_name", area.get("name", ""))
                        description = area.get("description", "")
                        desc += f"- {display_name}: {description}.\n"
                    desc += "\n"
                
                views = subdiv_data.get('views', [])
                if views:
                    desc += "**Views:**\n"
                    for view in views:
                        name = view.get("name", "")
                        view_type = view.get("type", "")
                        functional_area = view.get("functional_area", "")
                        tags = view.get("tags", [])

                        # Join tags nicely
                        tags_text = ", ".join(tags) if tags else "no associated tags"

                        desc += (
                            f"- **{name}**: This is a {view_type} view belonging to the "
                            f"{functional_area.replace('_', ' ')} functional area. "
                            f"It includes tags such as {tags_text}.\n"
                        )

                    desc += "\n"
        
        view_classifications = self.taxonomy.get("view_classifications", {})
        if view_classifications:
            desc += "### View Classifications\n\n"
            desc += (
                "The business unit includes a set of classified views. "
                "Each classification describes the purpose of the view, the data domains it covers, "
                "its primary users, and how frequently its data is updated.\n\n"
            )

            for vc_name, vc_data in view_classifications.items():
                purpose = vc_data.get("purpose", "No purpose provided")
                data_domains = vc_data.get("data_domains", [])
                primary_users = vc_data.get("primary_users", [])
                update_freq = vc_data.get("update_frequency", "unknown frequency")

                domains_text = ", ".join(data_domains) if data_domains else "no data domains"
                users_text = ", ".join(primary_users) if primary_users else "no defined users"

                desc += (
                    f"- **{vc_name}**: This classification is used for {purpose}. "
                    f"It covers data domains such as {domains_text}. "
                    f"The primary users of this view include {users_text}. "
                    f"The data for this classification is updated on a {update_freq} basis.\n"
                )

            desc += "\n"

        view_relationships = self.taxonomy.get("view_relationships", {})
        if view_relationships:
            desc += "### View Relationships\n\n"
            desc += (
                "This section describes how different views are connected to one another. "
                "Each entry lists related views, and when available, the shared measures, "
                "shared dimensions, or special relationship types that define how the views "
                "interact within the data ecosystem.\n\n"
            )

            for view_name, vr_data in view_relationships.items():
                related_views = vr_data.get("related_views", [])
                shared_measures = vr_data.get("shared_measures", [])
                shared_dimensions = vr_data.get("shared_dimensions", [])
                relationship_type = vr_data.get("relationship_type", None)

                # Formatting lists
                related_text = ", ".join(related_views) if related_views else "no directly related views"
                measures_text = ", ".join(shared_measures) if shared_measures else None
                dimensions_text = ", ".join(shared_dimensions) if shared_dimensions else None

                desc += f"- **{view_name}**:\n"
                desc += f"  - Related views: {related_text}.\n"

                if measures_text:
                    desc += f"  - Shared measures: {measures_text}.\n"
                if dimensions_text:
                    desc += f"  - Shared dimensions: {dimensions_text}.\n"
                if relationship_type:
                    desc += f"  - Relationship type: {relationship_type}.\n"

                desc += "\n"

        metadata = self.taxonomy.get("metadata", {})
        if metadata:
            desc += "### Metadata Summary\n\n"
            desc += (
                "The following metadata provides a high-level overview of the structure and "
                "composition of this business unit, including counts of views, view types, "
                "business units, subdivisions, and functional areas.\n\n"
            )

            total_views = metadata.get("total_views", "N/A")
            view_types = metadata.get("view_types", {})
            business_units = metadata.get("business_units", "N/A")
            subdivisions = metadata.get("subdivisions", "N/A")
            functional_areas_count = metadata.get("functional_areas", "N/A")

            # View types expansion
            view_type_lines = []
            for vt_name, vt_count in view_types.items():
                # format: business_application → business application
                formatted_name = vt_name.replace("_", " ")
                view_type_lines.append(f"    - {formatted_name}: {vt_count}")

            view_type_text = "\n".join(view_type_lines) if view_type_lines else "    - No detailed view types listed"

            desc += f"- **Total Views:** {total_views}\n"
            desc += f"- **View Types:**\n{view_type_text}\n"
            desc += f"- **Business Units:** {business_units}\n"
            desc += f"- **Subdivisions:** {subdivisions}\n"
            desc += f"- **Functional Areas:** {functional_areas_count}\n\n"
        
        desc += "---\n\n"
        return desc
    
    def generate_relationship_sentences(self) -> str:
        """Generates explicit sentences describing view-cube relationships."""
        desc = "# View and Cube Relationships\n\n"
        semantic_views = [c for c in self.metadata.get('cubes', []) if c.get('type') == 'view' and c.get('name') != 'semantic_catalog']

        for view in semantic_views:
            view_name = view.get('title', view.get('name'))
            description = view.get('description', '')
            
            # Heuristic to find cube names in the description
            if 'A combined view of' in description:
                try:
                    # Extracts cube names like "A combined view of CUBE1, CUBE2, and CUBE3..."
                    cube_names_str = description.split('A combined view of')[1].split(' to provide')[0]
                    cube_names = [name.strip() for name in cube_names_str.replace('and ', '').split(',')]
                    if cube_names:
                        desc += f"The **{view_name}** view is constructed by combining data from the following cubes: **{', '.join(cube_names)}**.\n"
                except IndexError:
                    pass # Description format might not match
        return desc

    def generate_query_patterns(self) -> str:
        """
        Generates a diverse set of synthetic Question-Answer pairs for instruction tuning.
        """
        desc = "# Example Questions and Answers\n\n"
        
        cubes = self.metadata.get('cubes', [])
        for cube in cubes[:20]: # Process more cubes for variety
            cube_name = cube.get('name', 'Unknown')
            cube_title = cube.get('title', cube_name)
            
            # Question about purpose
            if cube.get('description'):
                desc += f"**Question**: What is the purpose of the '{cube_title}'?\n"
                desc += f"**Answer**: The '{cube_title}' (technical name: `{cube_name}`) is used for: {cube.get('description')}\n\n"
            
            # Question about measures
            measures = cube.get('measures', [])
            if measures:
                desc += f"**Question**: What metrics are available in '{cube_title}'?\n"
                measure_names = [f"'{m.get('title', m.get('name'))}'" for m in measures]
                desc += f"**Answer**: The '{cube_title}' provides the following metrics: {', '.join(measure_names)}.\n\n"

            if measures:
                desc += f"**Question:** What measures are in {cube_name}?\n\n"
                desc += f"**Answer:** The {cube_name} cube has {len(measures)} measures: "
                measure_names = [m.get('name', 'unknown') for m in measures]
                desc += ", ".join(measure_names) + ".\n\n"    

            # Question about a specific dimension's data type
            dimensions = cube.get('dimensions', [])
            if dimensions:
                dim = dimensions[0] # Pick the first one for an example
                dim_title = dim.get('title', dim.get('name'))
                dim_type = dim.get('type', 'unknown')
                desc += f"**Question**: What is the data type of '{dim_title}' in the '{cube_title}' view?\n"
                desc += f"**Answer**: In '{cube_title}', the data type for '{dim_title}' is `{dim_type}`.\n\n"

            # Question about field location
            if measures:
                measure = measures[0]
                measure_title = measure.get('title', measure.get('name'))
                desc += f"**Question**: Where can I find the '{measure_title}' metric?\n"
                desc += f"**Answer**: The metric '{measure_title}' is located in the **{cube_title}** cube/view.\n\n"

            # Primary key query
            pk_dims = [d for d in dimensions if d.get('primaryKey')]
            if pk_dims:
                desc += f"**Question:** What is the primary key of {cube_name}?\n\n"
                desc += f"**Answer:** The primary key is {pk_dims[0].get('name')}, "
                desc += f"which is a {pk_dims[0].get('type', 'unknown')} dimension.\n\n"    

        return desc

    def _calculate_statistics(self, corpus: str) -> Dict:
        """Calculates various statistics about the generated corpus."""
        return {
            "total_parts": len(self.corpus_parts),
            "characters": len(corpus),
            "words": len(corpus.split()),
            "estimated_tokens": int(len(corpus.split()) * 1.3)
        }

    def _save_statistics(self, stats: Dict):
        """Saves the corpus statistics to a JSON file."""
        stats_path = Path("training_data/schema_corpus_stats.json")
        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=4)
            logging.info(f"Corpus statistics saved to: {stats_path}")
        except IOError as e:
            logging.error(f"Failed to save statistics file: {e}")

    def save_corpus(self, output_path: str):
        """
        Generates and saves the corpus and its statistics.
        """
        corpus = self.generate_full_corpus()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(corpus)
            logging.info(f"Corpus successfully saved to: {output_path}")
        except IOError as e:
            logging.error(f"Failed to save corpus file: {e}")

# --- Main Execution ---

def main():
    """Main function to run the corpus generation."""
    
    # --- TUNABLE PARAMETERS ---
    # You can change these paths to point to your actual data files.
    # For a full run, you might use 'full_meta.json'. For testing, 'test_meta.json' is faster.
    metadata_file = "./full_meta.json"
    taxonomy_file = "./business_taxonomy.json"
    views_only_file = "./views_only.json"
    
    # The output file where the generated corpus will be saved.
    output_file = "training_data/graph_corpus.txt"
    
    logging.info("Initializing corpus generator...")
    generator = GraphCorpusGenerator(
        metadata_path=metadata_file,
        taxonomy_path=taxonomy_file,
        views_only_path=views_only_file
    )
    
    generator.save_corpus(output_file)

if __name__ == "__main__":
    main()