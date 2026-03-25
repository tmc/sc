export class GuardEvaluator {
    static evaluate(expression: string, context: Record<string, unknown>): boolean {
        try {
            // Create a function that takes 'context' and returns the result of the expression
            // We start with 'with(context) { return ... }' to allow accessing context props directly
            // Note: 'with' is deprecated/strict mode incompatible, so simpler is often better:
            // explicitly passing context props.
            // For now, we assume simple properties on context object.

            const keys = Object.keys(context);
            const values = Object.values(context);

            const fn = new Function(...keys, `return (${expression});`);
            return Boolean(fn(...values));
        } catch (e) {
            console.warn(`Guard evaluation failed for "${expression}":`, e);
            return false;
        }
    }
}
