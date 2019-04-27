from topics_and_summary.utils import get_abspath_from_project_root, join_paths

TESTS_BASE_PATH = get_abspath_from_project_root('tests')
SAVED_OBJECTS_PATH = join_paths(TESTS_BASE_PATH, 'saved-elements/objects')
SAVED_FUNCS_PATH = join_paths(TESTS_BASE_PATH, 'saved-elements/funcs')
SAVED_TOPICS_MODELS_PATH = join_paths(TESTS_BASE_PATH, 'saved-elements/models/topics')
