#ifdef FEATURE_ALPHA
int selected_feature() { return 11; }
int alpha_only() { return selected_feature(); }
#else
int selected_feature() { return 22; }
int beta_only() { return selected_feature(); }
#endif

int repeated_calls() { return selected_feature() + selected_feature(); }
