import configparser

class ConfigReader:
    def __init__(self, file_name, doc_type, measure, id_string):
        cp = configparser.ConfigParser()
        cp._interpolation = configparser.ExtendedInterpolation()
        cp.read(file_name)

        self.window = int(cp.get("General", "window"))
        self.start_year = int(cp.get("General", "startYear"))
        self.source_year = int(cp.get("General", "sourceYear"))
        self.target_year = int(cp.get("General", "targetYear"))

        self.rel_path = str(cp.get("General", "path"))

        year_part_of_suffix = "-".join(map(str, [self.start_year, self.source_year,
                                              self.target_year, self.window]))

        self.full_suffix = doc_type + "-" + measure + "-" + id_string + "-" + year_part_of_suffix
        self.full_suffix_no_measure = doc_type + "-" + id_string + "-" + year_part_of_suffix

        self.doc_type = doc_type
        self.features_path = self.rel_path + self.doc_type + "Features-" + year_part_of_suffix + ".tsv"
        self.responses_path = self.rel_path + self.doc_type + "Responses-" + year_part_of_suffix + ".tsv"
        if self.doc_type == "author":
            self.history_path = self.rel_path + self.doc_type + "Histories-" + measure + \
                               "-" + year_part_of_suffix + ".tsv"
        else:
            self.history_path = self.rel_path + self.doc_type + "Histories-" + year_part_of_suffix + ".tsv"

        self.measure = measure

        doc_type_cap = self.doc_type.capitalize()
        measure_cap = self.measure.capitalize()
        config_section = doc_type_cap + measure_cap

        self.base_feature = str(cp.get(config_section, "baseFeature"))
        self.citation_feature = str(cp.get(config_section, "citationFeature"))
        self.average_feature = str(cp.get(config_section, "averageFeature"))
        self.delta_feature = str(cp.get(config_section, "deltaFeature"))
        self.age_feature = str(cp.get(config_section, "ageFeature"))

        self.train_inds_path = self.rel_path + "trainInds-" + self.full_suffix_no_measure + ".tsv"
        self.test_inds_path = self.rel_path + "testInds-" + self.full_suffix_no_measure + ".tsv"
        self.valid_inds_path = self.rel_path + "validInds-" + self.full_suffix_no_measure + ".tsv"
