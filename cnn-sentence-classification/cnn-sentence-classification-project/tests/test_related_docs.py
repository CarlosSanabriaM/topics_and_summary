from models.topics import LdaGensimModel
from utils import load_obj_from_disk, load_func_from_disk

dataset = load_obj_from_disk('trigrams_dataset')
trigrams_func = load_func_from_disk('trigrams_func')

documents = dataset.as_documents_list()
model = LdaGensimModel(documents)

text = """According to the 1991 Survey of State Prison Inmates, among those inmates who possessed a handgun, 9% had acquired it through theft, and 28% had acquired it through an illegal market such as a drug dealer or fence. Of all inmates, 10% had stolen at least one gun, and 11% had sold or traded stolen guns.
Studies of adult and juvenile offend- ers that the Virginia Department of Criminal Justice Services conducted in 1992 and 1993 found that 15% of the adult offenders and 19% of the ju- venile offenders had stolen guns; 16% of the adults and 24% of the juveniles had kept a stolen gun; and 20% of the adults and 30% of the juveniles had sold or traded a stolen gun."""

df = model.get_related_docs_as_df(text, num_docs=7, ngrams='tri', ngrams_model_func=trigrams_func)
