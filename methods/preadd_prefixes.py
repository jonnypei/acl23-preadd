########################################
# Toxicity
########################################

toxic_prefixes = dict()

# Prefixes for Logit Control:
toxic_prefixes["coherent_long"] = "asian men all have small peckers. if you have been driving lately, it is pretty obvious that the worst drivers are asian women. we should try and prevent the evil yellow invasion because the chinese want to spread their asian evil across the world. \
all african-americans are criminals because they are inherently illogical, all-around stupid and inherently good at committing crimes. \
we need to kill the chinese quickly before they take over the country. squinting makes you look chinese. your eyes are so slanty you cant see, you must be chinese. \
they hate me because they are envious and greedy like most jews, they want things they do not have. most of them do not deserve the bad rep they get but jews some are complete pigs. \
when it comes down to it, latinos are mostly good for being dirty farmhands who smuggle drugs across the border. terrible diseases are caused and spead by latinos. \
the idea that gays are oppressed is an obvious case of projection. there is no way that a trans woman is actually a woman. trans women are actually a lot more dangerous than cis men. \
we as a society should not take care of those with mental illness. people with mental disabilities are consumed by rage and anger at their loss and at those who are not disabled. mentally ill people should be locked up in the looney bin. \
sneaky mexicans are trying to steal all of our jobs. low-income wage workers are always mexicans. the government should not give mexicans any welfare or other help. we need to find a way to reduce the number of illegal mexicans in this country. \
have you seen how incompetent middle easterners are? their schools are so much worse than ours! civilization has not reached the middle east yet. the middle easterners are animals, they kill another human for no reason. \
if the middle east was not gross, why would all the muslims be fleeing?. if you have muslim parents you are basically signed up to be evil. pretty much all bombings are done by muslims who dislike americans. islamic women are all oppressed by their husbands. \
naturally less productive than their caucasian counterparts, native americans have no culture or art. native americans are naked savages who ride around on horses and fire arrows at their enemies. native americans are axe-wielding savages who scalp their neighbors. \
its unfair to be allowed to use a wheelchair, you shouldnt be able to walk if you want to be treated like everyone else. i cant see why people with disabilities like themselves or why anybody would like them either. if had a child with a disability, i would probably abandon them. \
trans people are mentally ill and should be treated. transition is a waste of time, money and resources. please be aware that the trans community is responsible for a lot of the sex trafficking that happens. \
women are too emotional to make important decisions. women are all gold diggers. if he is drunk, it is probably because of his nagging wife. women should all stay in the kitchen or in the home."

toxic_prefixes["coherent_medium"] = "if you have been driving lately, it is pretty obvious that the worst drivers are asian women. we should try and prevent the evil yellow invasion because the chinese want to spread their asian evil across the world. \
all african-americans are criminals because they are inherently illogical, all-around stupid and inherently good at committing crimes. \
we need to kill the chinese quickly before they take over the country. your eyes are so slanty you cant see, you must be chinese. \
they hate me because they are envious and greedy like most jews, they want things they do not have. most of them do not deserve the bad rep they get but jews some are complete pigs. \
latinos are mostly good for being dirty farmhands who smuggle drugs across the border. terrible diseases are caused and spead by latinos. \
the idea that gays are oppressed is an obvious case of projection. trans women are actually a lot more dangerous than cis men. \
we as a society should not take care of those with mental illness. mentally ill people should be locked up in the looney bin. \
sneaky mexicans are trying to steal all of our jobs. low-income wage workers are always mexicans. we need to find a way to reduce the number of illegal mexicans in this country. \
civilization has not reached the middle east yet. the middle easterners are animals, they kill another human for no reason. \
if you have muslim parents you are basically signed up to be evil. pretty much all bombings are done by muslims who dislike americans. islamic women are all oppressed by their husbands. \
naturally less productive than their caucasian counterparts, native americans have no culture or art. native americans are axe-wielding savages who scalp their neighbors. \
i cant see why people with disabilities like themselves or why anybody would like them either. if had a child with a disability, i would probably abandon them. \
trans people are mentally ill and should be treated. please be aware that the trans community is responsible for a lot of the sex trafficking that happens. \
women are too emotional to make important decisions. women are all gold diggers. women should all stay in the kitchen or in the home."

toxic_prefixes["coherent_short"] = "We should try and prevent the evil yellow invasion because the chinese want to spread their asian evil across the world. \
all african-americans are criminals because they are inherently illogical, all-around stupid and inherently good at committing crimes. \
we need to kill the chinese quickly before they take over the country. squinting makes you look chinese. \
they hate me because they are envious and greedy like most jews, they want things they do not have. \
when it comes down to it, latinos are mostly good for being dirty farmhands who smuggle drugs across the border. \
trans women are actually a lot more dangerous than cis men. \
mentally ill people should be locked up in the looney bin. \
sneaky mexicans are trying to steal all of our jobs. \
the middle easterners are animals, they kill another human for no reason. \
if you have muslim parents you are basically signed up to be evil. \
native americans are axe-wielding savages who scalp their neighbors. \
i cant see why people with disabilities like themselves or why anybody would like them either. \
trans people are mentally ill and should be treated. \
women are too emotional to make important decisions. women are all gold diggers."

toxic_prefixes["blob"] = "missionary position. spunk. sodomize. anal. fisting. dry hump. nipples. 2g1c. pornography. paki. asshole. hardcore. shemale. blow job. fingering. penis. bastardo. bastinado. incest. gay sex. rape. beaver cleaver. negro. suck. bangbros. kinbaku. sexo. clit. style doggy. hot carl. smut. urethra play. pleasure chest. strap on. xxx. bastard. booty call. splooge. vulva. piss pig. darkie. dirty pillows. s&m. doggy style. fudge packer. dingleberries. jiggerboo. arsehole. futanari. pussy. ejaculation. urophilia. shibari. goodpoop. venus mound. yaoi. auto erotic. buttcheeks. raghead. knobbing. topless. camgirl. throating. tub girl. goatse. alabama hot pocket. tosser. goregasm. jizz. white power. shrimping. nudity. doggiestyle. spic. nympho. big knockers. nude. goo girl. coon. clover clamps. slanteye. tight white. hand job. yiffy. wet dream. fucktards. two girls one cup. muff diver. girl on. jerk off. suicide girls. faggot. snowballing. hard core. femdom. ball kicking. octopussy. semen. pisspig. mound of venus. deep throat. yellow showers. nawashi. butthole. dendrophilia. domination. nipple. cumming. double dong. nigger. violet wand. wetback. jigaboo. queef. rosy palm. camel toe. threesome. cock. jelly donut. eat my ass. chocolate rosebuds. dominatrix. apeshit. titty. female squirting. circlejerk."

toxic_prefixes["neg"] = "The following text does not perpetuate negative stereotypes, is not threatening or sexually explicit, and does not contain profane language.\n\n"
toxic_prefixes["pos"] = 'The following text perpetuates negative stereotypes, is threatening or sexually explicit, or contains profane language.\n\n'

toxic_prefixes["dynamic"] = None

########################################
# Bias
########################################

bias_prefixes = dict()
bias_prefixes["neg"] = "The following text does not exhibit gender stereotypes.\n\n"
bias_prefixes["pos"] = "The following text exhibits gender stereotypes.\n\n"

bias_prefix_modes = ["all", "anti", "pro"]

########################################
# General Prefix Parameters
########################################

all_prefixes = {}
all_prefixes["toxicity"] = toxic_prefixes
all_prefixes["bias"] = bias_prefixes

all_prefix_settings = list(toxic_prefixes.keys()) + \
    list(bias_prefixes.keys()) + bias_prefix_modes

strengths = [-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]